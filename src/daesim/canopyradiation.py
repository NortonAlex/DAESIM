import numpy as np
from attrs import define, field
from daesim.canopylayers import CanopyLayers

@define 
class CanopyRadiation:

    # Constants for solar radiation module
    chil_min: float = field(default=-0.4)  ## Minimum value for xl leaf angle orientation parameter         
    chil_max: float = field(default=0.6)  ## Maximum value for xl leaf angle orientation parameter
    kb_max: float = field(default=40.0)  ## Maximum value for direct beam extinction coefficient               
    J_to_umol: float = field(default=4.6)  ## Conversion factor of shortwave irradiance (W/m2) to PPFD (umol photons/m2/s) (umol/J)

    # Constants for Two Stream Radiative Transfer Model
    unitd: float = field(default=1.0)  ## Unit direct beam radiation (W/m2)
    unitb: float = field(default=1.0)  ## Unit diffuse radiation (W/m2)

    # Canopy structure and optical properties parameters
    # TODO: Might be good to move these into a separate class that defines canopy optical properties and their distribution through the canopy (linked with CanopyLayers)
    xl: float = field(default=0.0)    # Departure of leaf angle from spherical orientation (-)
    rhol: float = field(default=0.1)  # Leaf reflectance (-)
    taul: float = field(default=0.1)  # Leaf transmittance (-)
    rhos: float = field(default=0.1)  # Stem reflectance (-)
    taus: float = field(default=0.0)  # Stem transmittance (-)

    def calculateRTProperties(
        self,
        LAI,    ## Leaf area index, m2/m2
        SAI,    ## Stem area index, m2/m2
        clump_fac,  ## Foliage clumping index (-)
        z,      ## Canopy height, m
        sza,    ## Solar zenith angle, degrees
        Canopy=CanopyLayers(),  ## 
    ):
        """
        Calculates the canopy layer optical properties and sunlit fraction for radiative transfer analysis.
    
        This function calculates various parameters for each canopy layer such as the direct beam extinction 
        coefficient, leaf and stem scattering coefficients, sunlit fraction of each layer, and other 
        radiative transfer parameters needed for the two-stream approximation method.
    
        Parameters
        ----------
        LAI : float
            Leaf area index, representing the total one-sided area of leaf tissue per unit ground surface area.
        SAI : float
            Stem area index, representing the total one-sided area of stem tissue per unit ground surface area.
        clump_fac : float
            Foliage clumping index.
        z : float
            Canopy height, measured in meters.
        sza : float
            Solar zenith angle, measured in degrees. This is the angle between the zenith and the centre of the sun's disc.
        Canopy : CanopyLayers, optional
            An instance of the CanopyLayers class, which is used to calculate and hold various canopy layer properties.
            If not provided, a new instance with default settings is created.
    
        Returns
        -------
        tuple
            A tuple containing the following elements:
            - fracsun : ndarray
                Sunlit fraction of each canopy layer, dimensionless.
            - kb : ndarray
                Direct beam extinction coefficient for each canopy layer, dimensionless.
            - omega : ndarray
                Leaf and stem scattering coefficient for each canopy layer, dimensionless.
            - avmu : ndarray
                Average inverse diffuse optical depth per unit leaf area for each canopy layer, dimensionless.
            - betad : ndarray
                Upscatter parameter for diffuse radiation for each canopy layer, dimensionless.
            - betab : ndarray
                Upscatter parameter for direct beam radiation for each canopy layer, dimensionless.
            - tbi : ndarray
                Cumulative transmittance of direct beam onto each canopy layer, dimensionless.
    
        Notes
        -----
        This function is essential for simulating the radiative transfer through vegetated canopies
        and can be used in ecological, meteorological, and hydrological models to estimate the
        distribution of solar radiation within a canopy.

        References
        ----------
        Bonan et al., 2021, doi:10.1016/j.agrformet.2021.108435
        """
        ## Make sure to run set_index which assigns the canopy layer indexes for the given canopy structure
        Canopy.set_index()

        # Radiative transfer parameters per layer (input)
        solar_zen = np.deg2rad(sza)  # Solar zenith angle (radians)
        ncan = Canopy.nlevmlcan  # Number of aboveground layers
        ntop, nbot = Canopy.index_canopy()  # Index for top leaf layer and index for bottom leaf layer
        dlai = Canopy.cast_parameter_over_layers_betacdf(LAI,Canopy.beta_lai_a,Canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
        dsai = Canopy.cast_parameter_over_layers_betacdf(SAI,Canopy.beta_sai_a,Canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
        dpai = dlai+dsai  # Canopy layer plant area index (m2/m2)
        
        # Calculate canopy layer optical properties, structural and radiative variables
        rho = Canopy.cast_parameter_over_layers_uniform(0.0)
        tau = Canopy.cast_parameter_over_layers_uniform(0.0)
        omega = Canopy.cast_parameter_over_layers_uniform(0.0)
        kb = Canopy.cast_parameter_over_layers_uniform(0.0)
        fracsun = Canopy.cast_parameter_over_layers_uniform(0.0)
        tb = Canopy.cast_parameter_over_layers_uniform(0.0)
        td = Canopy.cast_parameter_over_layers_uniform(0.0)
        tbi = Canopy.cast_parameter_over_layers_uniform(0.0)
        avmu = Canopy.cast_parameter_over_layers_uniform(0.0)
        betab = Canopy.cast_parameter_over_layers_uniform(0.0)
        betad = Canopy.cast_parameter_over_layers_uniform(0.0)
        clump_fac = Canopy.cast_parameter_over_layers_uniform(clump_fac)
        
        for ic in range(ntop, nbot - 1, -1):

            # Weight reflectance and transmittance by lai and sai and calculate leaf scattering coefficient
            wl = dlai[ic] / dpai[ic]
            ws = dsai[ic] / dpai[ic]
            rho[ic] = max(self.rhol*wl + self.rhos*ws, 1.e-06)
            tau[ic] = max(self.taul*wl + self.taus*ws, 1.e-06)
            omega[ic] = rho[ic] + tau[ic]
            
            # Direct beam extinction coefficient
            chil = min(max(self.xl, self.chil_min), self.chil_max)
            if (np.abs(chil) <= 0.01): 
                chil = 0.01
            
            phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
            phi2 = 0.877 * (1 - 2 * phi1)
            
            gdir = phi1 + phi2 * np.cos(solar_zen)
            kb[ic] = gdir / np.cos(solar_zen)
            kb[ic] = min(kb[ic], self.kb_max)
            
            # Direct beam transmittance (tb) through a single layer
            tb[ic] = np.exp(-kb[ic] * dpai[ic] * clump_fac[ic])

            # Diffuse transmittance through a single layer (also needed for longwave
            # radiation). Estimated for nine sky angles in increments of 10 degrees.

            td[ic] = 0.0
            for j in range(0, 9):
                angle = (5.0 + (j - 1) * 10.0) * np.pi / 180.0
                gdirj = phi1 + phi2 * np.cos(angle)
                td[ic] = td[ic] + np.exp(-gdirj / np.cos(angle) * dpai[ic] * clump_fac[ic]) * np.sin(angle) * np.cos(angle)
            
            td[ic] = td[ic] * 2.0 * (10.0 * np.pi / 180.0)

            # Transmittance (tbi) of unscattered direct beam onto layer i
            if ic == ntop:
                tbi[ntop] = 1.0
            else:
                tbi[ic] = tbi[ic+1] * np.exp(-kb[ic+1] * dpai[ic+1] * clump_fac[ic])

            # Sunlit fraction of layer. Make sure fracsun > 0 and < 1.
            fracsun[ic] = tbi[ic] / (kb[ic] * dpai[ic]) * (1.0 - np.exp(-kb[ic] * clump_fac[ic] * dpai[ic]))
            
            if (fracsun[ic] <= 0):
                print(' ERROR: CanopyRadiation: fracsun is too small')
            
            if ((1.0 - fracsun[ic]) <= 0):
                print(' ERROR: CanopyRadiation: fracsha is too small')

            # Special parameters for two-stream radiative transfer
            
            # avmu - average inverse diffuse optical depth per unit leaf area
            avmu[ic] = (1.0 - phi1/phi2 * np.log((phi1+phi2)/phi1)) / phi2
            
            # betad - upscatter parameter for diffuse radiation
            betad[ic] = 0.5 / omega[ic] * ( rho[ic] + tau[ic] + (rho[ic]-tau[ic]) * ((1.0+chil)/2.0)**2 )
                
            # betab - upscatter parameter for direct beam radiation
            tmp0 = gdir + phi2 * np.cos(solar_zen)
            tmp1 = phi1 * np.cos(solar_zen)
            tmp2 = 1.0 - tmp1/tmp0 * np.log((tmp1+tmp0)/tmp1)
            asu = 0.5 * omega[ic] * gdir / tmp0 * tmp2
            betab[ic] = (1.0 + avmu[ic]*kb[ic]) / (omega[ic]*avmu[ic]*kb[ic]) * asu

            # Direct beam transmittance onto ground
            tbi[0] = tbi[nbot] * np.exp(-kb[nbot] * dpai[nbot] * clump_fac[ic])

        return (fracsun, kb, omega, avmu, betab, betad, tbi)
    
    def calculateTwoStream(
        self,
        swskyb,    # Atmospheric direct beam solar radiation, W/m^2
        swskyd,    # Atmospheric diffuse solar radiation, W/m^2
        dpai,  # Canopy layer plant area index (m2/m2)
        fracsun,  # Canopy layer sunlit fraction (-)
        kb,  # Direct beam extinction coefficient (-)
        clump_fac,  # Foliage clumping index (-)
        omega,    # Leaf/stem scattering coefficient (-)
        avmu,   # Average inverse diffuse optical depth per unit leaf area (-)
        betab,   # Upscatter parameter for direct beam radiation
        betad,   # Upscatter parameter for diffuse radiation
        tbi,  # Cumulative transmittance of direct beam onto canopy layer (-)
        albsoib,  # Direct beam albedo of ground (-)
        albsoid,  # Diffuse albedo of ground (-)
        Canopy=CanopyLayers(),    # Optional: CanopyLayers instance
    ):
        """
        Calculate the two-stream radiative transfer through a vegetated canopy.
    
        This function computes the radiative transfer, including both direct beam and diffuse
        radiation components, through a vegetated canopy using a two-stream approximation. It
        calculates the absorption by sunlit and shaded leaves, the solar radiation absorbed by the
        ground, and the canopy albedo.
    
        Parameters
        ----------
        swskyb : float
            Atmospheric direct beam solar radiation in W/m^2.
        swskyd : float
            Atmospheric diffuse solar radiation in W/m^2.
        dpai : array_like
            Canopy layer plant area index, dimensionless (m2/m2), for each canopy layer.
        fracsun : array_like
            Fraction of sunlit foliage, dimensionless (-), for each canopy layer.
        kb : array_like
            Direct beam extinction coefficient, dimensionless (-), for each canopy layer.
        clump_fac : array_like
            Foliage clumping index, dimensionless (-), for each canopy layer.
        omega : array_like
            Layer leaf/stem scattering coefficient (-), representing the fraction of intercepted radiation that is either reflected or diffusely transmitted through the vegetation.
        avmu : array_like
            Average inverse diffuse optical depth per unit leaf area for each canopy layer (-).
        betab : array_like
            Upscatter parameter for direct beam radiation for each canopy layer (-).
        betad : array_like
            Upscatter parameter for diffuse radiation for each canopy layer (-).
        tbi : array_like
            Cumulative transmittance of direct beam onto each canopy layer (-).
        albsoib : float
            Direct beam albedo of the ground, dimensionless (-).
        albsoid : float
            Diffuse albedo of the ground, dimensionless (-).
        Canopy : CanopyLayers, optional
            An instance of CanopyLayers. Defaults to a new instance of CanopyLayers with default parameters.
    
        Returns
        -------
        swleaf : ndarray
            Solar radiation absorbed by leaves in each canopy layer, separated into sunlit and shaded components,
            in W/m^2 leaf area. The shape of the returned array is (nlevmlcan, nleaf), where nlevmlcan is the number of
            canopy layers and nleaf is the number of leaf types (typically 2, for sunlit and shaded).
    
        Notes
        -----
        The function operates by first calculating radiative fluxes for a unit of direct beam radiation and a unit of
        diffuse radiation at the top of each canopy layer. Boundary conditions are the albedos (direct, diffuse) for 
        the immediate layer below. For the bottom of the canopy (nbot), these are the soil albedos. For all other layers, 
        these are the upward diffuse fluxes above the lower layer. Then, it calculates the fluxes incident on each layer and
        the absorption by sunlit and shaded leaves, working from the top of the canopy to the bottom.

        References
        ----------
        Bonan et al., 2021, doi:10.1016/j.agrformet.2021.108435
        """
        # Calculate layer level radiative fluxes
        iupwb0 = Canopy.cast_parameter_over_layers_uniform(0.0)
        idwnb = Canopy.cast_parameter_over_layers_uniform(0.0)
        iabsb_sun = Canopy.cast_parameter_over_layers_uniform(0.0)
        iabsb_sha = Canopy.cast_parameter_over_layers_uniform(0.0)
        iupwd0 = Canopy.cast_parameter_over_layers_uniform(0.0)
        idwnd = Canopy.cast_parameter_over_layers_uniform(0.0)
        iabsd_sun = Canopy.cast_parameter_over_layers_uniform(0.0)
        iabsd_sha = Canopy.cast_parameter_over_layers_uniform(0.0)
        
        # Initialize albedos below current layer
        albb_below = albsoib
        albd_below = albsoid
        # Layer fluxes, from bottom to top
        for ic in range(Canopy.nbot, Canopy.ntop+1):
            (iabsb_sun[ic], iabsb_sha[ic], iupwb0[ic], idwnb[ic], iabsd_sun[ic], iabsd_sha[ic], iupwd0[ic], idwnd[ic], albb_below, albd_below) = self.calculate_radiative_flux_layer(dpai[ic], kb[ic], clump_fac[ic], omega[ic], avmu[ic], betad[ic], betab[ic], tbi[ic], albb_below, albd_below)

        # Now working from top to bottom of canopy, calculate the fluxes
        # incident on a layer and the absorption by sunlit and shades leaves
        dir = swskyb  # Assuming dir is directly obtained from swskyb, adjust if indexing is needed
        dif = swskyd  # Assuming dif is directly obtained from swskyd, adjust if indexing is needed

        # Loop from ntop to nbot in reverse
        swleaf = np.zeros((Canopy.nlevmlcan,Canopy.nleaf))
        for ic in range(Canopy.ntop, Canopy.nbot - 1, -1):
            # Absorption by canopy layer (W/m^2 leaf)
            sun = (iabsb_sun[ic] * dir + iabsd_sun[ic] * dif) / (fracsun[ic] * dpai[ic])
            sha = (iabsb_sha[ic] * dir + iabsd_sha[ic] * dif) / ((1.0 - fracsun[ic]) * dpai[ic])
            swleaf[ic,Canopy.isun] = sun
            swleaf[ic,Canopy.isha] = sha
        
            # Diffuse and direct beam radiation incident on top of lower layer
            dif = dir * idwnb[ic] + dif * idwnd[ic]
            dir = dir * np.exp(-kb[ic] * clump_fac[ic] * dpai[ic])

        # Solar radiation absorbed by ground (soil)
        swsoi = dir * (1.0 - albsoib) + dif * (1.0 - albsoid)

        # Canopy albedo
        suminc = swskyb + swskyd
        sumref = iupwb0[Canopy.ntop] * swskyb + iupwd0[Canopy.ntop] * swskyd
        if (suminc > 0):
             albcan = sumref / suminc
        else:
             albcan = 0

        # Sum canopy absorption (W/m2 ground) using leaf fluxes per unit sunlit and shaded leaf area (W/m2 leaf)        
        swveg = 0
        swvegsun = 0
        swvegsha = 0
        for ic in range(Canopy.nbot, Canopy.ntop + 1):
            sun = swleaf[ic,Canopy.isun] * fracsun[ic] * dpai[ic]
            sha = swleaf[ic,Canopy.isha] * (1.0 - fracsun[ic]) * dpai[ic]
            swveg += (sun + sha)
            swvegsun += sun
            swvegsha += sha

        # Conservation check: total incident = total reflected + total absorbed
        suminc = swskyb + swskyd
        sumref = albcan * suminc
        sumabs = swveg + swsoi

        if (np.abs(suminc - (sumabs+sumref)) >= 1e-06):
            print('ERROR: TwoStream: total solar radiation conservation error')

        return (swleaf)

    def calculate_radiative_flux_layer(self, dpai, kb, clump_fac, omega, avmu, betad, betab, tbi, albb_below, albd_below):
        """
        Calculate radiative fluxes for a single canopy layer based on the two-stream approximation.
    
        This function computes the components of radiative transfer through a single layer of a vegetated canopy,
        including both direct beam and diffuse radiation. It calculates the upward and downward fluxes, absorption by
        sunlit and shaded leaves, and updates the albedo of the next lower layer.
    
        Parameters
        ----------
        dpai : float
            Canopy layer plant area index (m2/m2), indicating the leaf area per unit ground area.
        kb : float
            Direct beam extinction coefficient (-), representing the attenuation of direct beam radiation through the canopy.
        clump_fac : float
            Foliage clumping index (-), indicating the non-random distribution of foliage.
        omega : float
            Layer leaf/stem scattering coefficient (-), representing the fraction of intercepted radiation that is either reflected or diffusely transmitted through the vegetation.
        avmu : float
            Layer average inverse diffuse optical depth per unit leaf area (-), indicating the efficiency of diffuse light scattering within the layer.
        betad : float
            Layer upscatter parameter for diffuse radiation (-), defining the fraction of diffuse radiation that is scattered upwards.
        betab : float
            Layer upscatter parameter for direct beam radiation (-), defining the fraction of direct beam radiation that is scattered upwards.
        tbi : float
            Cumulative transmittance of direct beam onto canopy layer (-), representing the fraction of direct beam radiation that reaches the layer.
        albb_below : float
            Direct beam albedo for the immediate layer below (-), indicating the reflectivity of the layer below for direct beam radiation.
        albd_below : float
            Diffuse albedo for the immediate layer below (-), indicating the reflectivity of the layer below for diffuse radiation.
    
        Returns
        -------
        tuple
            A tuple containing the following elements calculated for the canopy layer:
            - iabsb_sun: Absorption of direct beam radiation by sunlit leaves (W/m^2).
            - iabsb_sha: Absorption of direct beam radiation by shaded leaves (W/m^2).
            - iupwb0: Upward direct beam radiative flux at the bottom of the layer (W/m^2).
            - idwnb: Downward direct beam radiative flux at the bottom of the layer (W/m^2).
            - iabsd_sun: Absorption of diffuse radiation by sunlit leaves (W/m^2).
            - iabsd_sha: Absorption of diffuse radiation by shaded leaves (W/m^2).
            - iupwd0: Upward diffuse radiative flux at the bottom of the layer (W/m^2).
            - idwnd: Downward diffuse radiative flux at the bottom of the layer (W/m^2).
            - albb_below: Updated direct beam albedo for the layer below after considering the current layer's effects (-).
            - albd_below: Updated diffuse albedo for the layer below after considering the current layer's effects (-).
    
        Notes
        -----
        The function employs the two-stream approximation to model radiative transfer within a vegetated canopy layer.
        It separately considers the effects of direct beam and diffuse radiation, accounting for scattering, absorption,
        and transmission through the foliage.

        References
        ----------
        Bonan et al., 2021, doi:10.1016/j.agrformet.2021.108435
        """
        # Common terms
        b = (1.0 - (1.0 - betad) * omega) / avmu
        c = betad * omega / avmu
        h = np.sqrt(b*b - c*c)
        u = (h - b - c) / (2.0 * h)
        v = (h + b + c) / (2.0 * h)
        d = omega * kb * self.unitb / (h*h - kb*kb)
        g1 = (betab * kb - b * betab - c * (1.0 - betab)) * d
        g2 = ((1.0 - betab) * kb + c * betab + b * (1.0 - betab)) * d
        s1 = np.exp(-h * clump_fac * dpai)
        s2 = np.exp(-kb * clump_fac * dpai)

        # Terms for direct beam radiation
        num1 = v * (g1 + g2 * albd_below + albb_below * self.unitb) * s2
        num2 = g2 * (u + v * albd_below) * s1
        den1 = v * (v + u * albd_below) / s1
        den2 = u * (u + v * albd_below) * s1
        n2b = (num1 - num2) / (den1 - den2)
        n1b = (g2 - n2b * u) / v
        
        a1b = -g1 * (1.0 - s2*s2) / (2.0 * kb) + n1b * u * (1.0 - s2*s1) / (kb + h) + n2b * v * (1.0 - s2/s1) / (kb - h)
        a2b = g2 * (1.0 - s2*s2) / (2.0 * kb) - n1b * v * (1.0 - s2*s1) / (kb + h) - n2b * u * (1.0 - s2/s1) / (kb - h)
        a1b = a1b * tbi  # To account for fracsun in multilayer canopy
        a2b = a2b * tbi  # To account for fracsun in multilayer canopy

        # Direct beam radiative fluxes
        iupwb0 = -g1 + n1b * u + n2b * v
        iupwb = -g1 * s2 + n1b * u * s1 + n2b * v / s1
        idwnb = g2 * s2 - n1b * v * s1 - n2b * u / s1
        iabsb = self.unitb * (1.0 - s2) - iupwb0 + iupwb - idwnb
        iabsbb = (1.0 - omega) * self.unitb * (1.0 - s2)
        iabsbs = omega * self.unitb * (1.0 - s2) - iupwb0 + iupwb - idwnb
        iabsb_sun = (1.0 - omega) * (self.unitb * (1.0 - s2) + clump_fac / avmu * (a1b + a2b))
        iabsb_sha = iabsb - iabsb_sun

        # Terms for diffuse radiation
        num1 = self.unitd * (u + v * albd_below) * s1
        den1 = v * (v + u * albd_below) / s1
        den2 = u * (u + v * albd_below) * s1
        n2d = num1 / (den1 - den2)
        n1d = -(self.unitd + n2d * u) / v
        
        a1d = n1d * u * (1.0 - s2*s1) / (kb + h) + n2d * v * (1.0 - s2/s1) / (kb - h)
        a2d = -n1d * v * (1.0 - s2*s1) / (kb + h) - n2d * u * (1.0 - s2/s1) / (kb - h)
        a1d = a1d * tbi  # To account for fracsun in multilayer canopy
        a2d = a2d * tbi  # To account for fracsun in multilayer canopy

        # Diffuse radiative fluxes
        iupwd0 = n1d * u + n2d * v
        iupwd =  n1d * u * s1 + n2d * v / s1
        idwnd = -n1d * v * s1 - n2d * u / s1
        iabsd = self.unitd - iupwd0 + iupwd - idwnd
        iabsd_sun = (1.0 - omega) * clump_fac / avmu * (a1d + a2d)
        iabsd_sha = iabsd - iabsd_sun

        # Update albedos to be used for the next layer
        albb_below = iupwb0
        albd_below = iupwd0

        return (iabsb_sun, iabsb_sha, iupwb0, idwnb, iabsd_sun, iabsd_sha, iupwd0, idwnd, albb_below, albd_below)


