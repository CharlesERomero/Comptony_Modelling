import numpy as np
from astropy.io import fits
from astropy import wcs   # A slight variation...
import tSZ_spectrum as tsz
import kSZ_spectrum as ksz
import astropy.units as u
import astropy.constants as const
import scipy.constants as spconst
from astropy.cosmology import Planck15 as cosmo
import numerical_integration as ni

############################################################################
            
def inst_params(instrument):

    if instrument == "MUSTANG":
        fwhm1 = 8.7*u.arcsec  # arcseconds
        norm1 = 0.94          # normalization
        fwhm2 = 28.4*u.arcsec # arcseconds
        norm2 = 0.06          # normalization
        fwhm  = 9.0*u.arcsec
        smfw  = 10.0*u.arcsec
        freq  = 90.0*u.gigahertz # GHz
        FoV   = 42.0*u.arcsec #

    ### I don't use the double Guassian much. The only idea was to use it to
    ### get a better estimate of the beam volume, but we know that is variable.
    if instrument == "MUSTANG2":
        fwhm1 = 8.9*u.arcsec  # arcseconds
        norm1 = 0.97          # normalization
        fwhm2 = 25.0*u.arcsec # arcseconds
        norm2 = 0.03          # normalization
        fwhm  = 9.0*u.arcsec
        smfw  = 10.0*u.arcsec
        freq  = 90.0*u.gigahertz # GHz
        FoV   = 4.25*u.arcmin 
        
    if instrument == "NIKA":
        fwhm1 = 8.7*2.0*u.arcsec  # arcseconds
        norm1 = 0.94     # normalization
        fwhm2 = 28.4*2.0*u.arcsec # arcseconds
        norm2 = 0.06     # normalization
        fwhm  = 18.0*u.arcsec
        smfw  = 10.0*u.arcsec
        freq  = 150.0*u.gigahertz    # GHz
        FoV   = 2.15*u.arcmin 
        
    if instrument == "NIKA2":
        fwhm1 = 8.7*2.0*u.arcsec  # arcseconds
        norm1 = 0.94     # normalization
        fwhm2 = 28.4*2.0*u.arcsec # arcseconds
        norm2 = 0.06     # normalization
        fwhm  = 18.0*u.arcsec
        smfw  = 10.0*u.arcsec
        freq  = 150.0*u.gigahertz    # GHz
        FoV   = 6.5*u.arcmin 
        
    if instrument == "BOLOCAM":
        fwhm1 = 8.7*7.0*u.arcsec  # arcseconds
        norm1 = 0.94     # normalization
        fwhm2 = 28.4*7.0*u.arcsec # arcseconds
        norm2 = 0.06     # normalization
        fwhm  = 58.0*u.arcsec
        smfw  = 60.0*u.arcsec
        freq  = 140.0*u.gigahertz    # GHz
        FoV   = 8.0*u.arcmin * (u.arcmin).to("arcsec")
        
    if instrument == "ACT90":
        fwhm1 = 2.16*60.0*u.arcsec  # arcseconds
        norm1 = 1.0     # normalization
        fwhm2 = 1.0*u.arcsec # arcseconds
        norm2 = 0.00     # normalization
        fwhm  = 2.16*60.0*u.arcsec
        smfw  = 2.0*60.0*u.arcsec
        freq  = 97.0*u.gigahertz    # GHz
        FoV   = 60.0*u.arcmin #* (u.arcmin).to("arcsec")
        
    if instrument == "ACT150":
        fwhm1 = 1.3*60.0*u.arcsec  # arcseconds
        norm1 = 1.0     # normalization
        fwhm2 = 1.0*u.arcsec # arcseconds
        norm2 = 0.00     # normalization
        fwhm  = 1.3*60.0*u.arcsec
        smfw  = 1.2*60.0*u.arcsec
        freq  = 148.0*u.gigahertz    # GHz
        FoV   = 60.0*u.arcmin #* (u.arcmin).to("arcsec")
        

#    else:
#        fwhm1=9.0*u.arcsec ; norm1=1.0
#        fwhm2=30.0*u.arcsec ; norm2=0.0
#        fwhm = 9.0*u.arcsec ; smfw = 10.0*u.arcsec
#        freq = 90.0*u.gigahertz 
#        FoV   = 1.0*u.arcmin * (u.arcmin).to("arcsec")
#        
#    import pdb; pdb.set_trace()

    return fwhm1,norm1,fwhm2,norm2,fwhm,smfw,freq,FoV

def inst_bp(instrument,array="2"):
    """
    Returns a frequency and bandpass array
    
    Parameters
    __________
    instrument : MUSTANG, MUSTANG2, BOLOCAM, NIKA, or NIKA2
    currently only MUSTANG2 and NIKA2 are supported

    Returns
    -------
    -> farr   - The frequency array (in GHz)
    -> band   - the bandpass, with aperture efficiency applied
    """

    if instrument == "MUSTANG2" or instrument == "MUSTANG":
        srms = (300*u.um).to("m")        # surface RMS (microns)
        ### Reference: https://science.nrao.edu/facilities/gbt/proposing/GBTpg.pdf
        EA90 = 0.36   # Aperture efficiency at 90 GHz
        ### The beam efficiencies should be taken as 1.37* Aperture Efficiency
        R90  = np.exp(-4.0*np.pi*(srms/(const.c/(9.0e10*u.s**-1))).value)    #
        Gnot = EA90/R90                   # Unphysical, but see documentation...
        if instrument == "MUSTANG2":
            flow = 75.0   # GHz
            fhig = 105.0  # GHz
        else:
            flow = 82.5   # GHz
            fhig = 97.5  # GHz
            
        farr = np.arange(flow,fhig,1.0)  # frequency array.
        tran = farr*0.0 + 1.0            # Let the transmission be unity everywhere.
        Larr = const.c.value/(farr*1.0e9) # Keep calm and carry on.
        ### Old formula:
        #Ruze = Gnot * np.exp(-4.0*np.pi*(srms.value)/Larr)
        ### Correct formula: (10 April 2018)
        Ruze = Gnot * np.exp(-(4.0*np.pi*srms.value/Larr)**2)
        NRuz = Ruze / np.max(Ruze)        # Normalize it
        band = tran * Ruze                # Bandpass, with (unnormalized) Ruze efficiency
       
    if instrument == "NIKA2" or instrument == "NIKA":
        caldir='/home/romero/NIKA2/NIKA_SVN/Processing/Pipeline/Calibration/BP/'
        bpfile=caldir+'Transmission_2017_Jan_NIKA2_v1.fits'
        hdulist = fits.open(bpfile)

        if array == "1H":      # 1mm (260 GHz) array, Horizontal Polarization
            tbdata = hdulist[1].data # 1H
            freq = tbdata.field(0)
            tran = tbdata.field(1)
            erro = tbdata.field(2)
            atmt = tbdata.field(3)
            cfreq1h = np.sum(freq*tran)/np.sum(tran)
        
        if array == "1V":     # 1mm (260 GHz) array, Vertical Polarization
            tbdata = hdulist[2].data # 1V
            freq = tbdata.field(0)
            tran = tbdata.field(1)
            erro = tbdata.field(2)
            atmt = tbdata.field(3)
            cfreq1v = np.sum(freq*tran)/np.sum(tran)
        
        if array == "2":       # 2mm (150 GHz) array
            tbdata = hdulist[3].data # 2
            freq = tbdata.field(0)
            tran = tbdata.field(1)
            erro = tbdata.field(2)
            atmt = tbdata.field(3)
            cfreq2 = np.sum(freq*tran)/np.sum(tran)

        ### Trim the zero-frequency listing, if any.
        gi=np.where(freq > 0)
        freq = freq[gi]
        tran = tran[gi]
        erro = erro[gi]
        atmt = atmt[gi]
        
### Calculate Aperture efficiencies from information found at:
### http://www.iram.es/IRAMES/mainwiki/Iram30mEfficiencies
        Beff = 0.630         # at 210 GHz
        Aeff = Beff/1.27     # See text on webpage
        srms = (66.0*u.um).to("m")        # surface RMS (microns)
        R210 = np.exp(-4.0*np.pi*(srms/(const.c/(2.1e11*u.s**-1))).value)    #
        Gnot = Aeff/R210                   # Unphysical, but see documentation...

        Larr = const.c.value/(freq*1.0e9) # Keep calm and carry on.        
        Ruze = Gnot * np.exp(-4.0*np.pi*(srms.value)/Larr)
        NRuz = Ruze / np.max(Ruze)        # Normalize it
        band = tran * Ruze                # Bandpass, with (unnormalized) Ruze efficiency
        farr = freq
        
#########################################################################

    return band, farr
        
def get_sz_bp_conversions(temp,instrument,units='Jy/beam',array="2",inter=False,beta=1.0/300.0,
                          betaz=1.0/300.0,rel=True,quiet=False):

    szcv,szcu=get_sz_values()
    freq_conv = (szcv['planck'] *1.0e9)/(szcv['boltzmann']*szcv['tcmb'])
    temp_conv = 1.0/szcv['m_e_c2']
    bv = get_beamvolume(instrument)
#    fwhm1,norm1,fwhm2,norm2,fwhm,smfw,freq,FoV = inst_params(instrument)
    band, farr = inst_bp(instrument,array)
    fstep = np.median(farr - np.roll(farr,1))
    foo = np.where(band > 0)
    band=band[foo]
    farr=farr[foo]
############################################
    llt  = temp*temp_conv  # Lower limit on temperature (well, theta)
    ult  = temp*temp_conv  # Upper limit on temperature
    st   = temp            # Temperature (theta) step
    flow = np.min(farr)
 #   fhigh= np.ceil((np.max(farr)-flow)/fstep)*fstep+flow+fstep/100.0
    fhigh= np.max(farr)
    sx   = fstep*freq_conv
    llx  = flow*freq_conv
    nste = (fhigh-flow)/fstep
    ulx  = fhigh*freq_conv + sx/(2.0*nste)
    
#### The old way:
#    temparr,freqarr,conv = tsz.tSZ_conv_range(tlow,thigh,tstep,flow,fhigh,fstep)
#### The new way (for now):

    if inter == True:    
        tarr = np.arange(llt,ult,st)
        xarr = np.arange(llx,ulx,sx)
        T = tsz.itoh_2004_int(tarr,xarr)
    else:
        tarr, xarr, T = tsz.tSZ_conv(llt,llx,ult,ulx,st,sx)
        
    tarr, xarr, K = ksz.kSZ_conv(beta,betaz,llt,llx,ult,ulx,st,sx,rel=rel)
    
#    import pdb; pdb.set_trace()
### Let's check that the frequency spacing is close to what was given in the
### bandpass retreival:
    fq = np.abs(farr*freq_conv - xarr)/(farr*freq_conv)
    if np.max(fq) > 0.05:
        raise Exception("Frequency arrays differ significantly.")
## Else (implicit here):
    TY = T/tarr    # Divide by tarr (thetae) to get proper conversion units
    KY = K/tarr    # Divide by tarr (thetae) to get proper conversion units
    bT = np.sum(TY*band)/np.sum(band)  # Average over the bandpass
    bK = np.sum(KY*band)/np.sum(band)  # Average over the bandpass

    JypB = tsz.Jyperbeam_factors(bv)        # Jy per beam conversion factor, from y (bT)
    xavg = np.sum(xarr*band)/np.sum(band)   # Bandpass averaged frequency; should be a reasonable approximation.
    Kepy = tsz.TBright_factors(xavg)        # Kelvin conversion factor, from y (bT)
    
    tSZ_JyBeam_per_y = JypB * bT  # Just multiply by Compton y to get Delta I (tSZ)
    kSZ_JyBeam_per_t = JypB * bK  # Just multiply by tau (of electrons) to get Delta I (kSZ)
    tSZ_Kelvin_per_y = Kepy * bT  # Just multiply by Compton y to get Delta T (tSZ)
    kSZ_Kelvin_per_t = Kepy * bK  # Just multiply by tau (of electrons) to get Delta T (kSZ)

    if quiet == False:
        print 'Assuming a temperature of ',temp,' keV, we find the following.'
        print 'To go from Compton y to Jy/Beam, multiply by: ', tSZ_JyBeam_per_y
        print 'To go from tau to Jy/Beam (kSZ), multiply by: ', kSZ_JyBeam_per_t
        print 'To go from Compton y to Kelvin, multiply by: ', tSZ_Kelvin_per_y
        print 'To go from tau to Kelvin (kSZ), multiply by: ', kSZ_Kelvin_per_t
    
    if units == 'Kelvin':
        tSZ_return = tSZ_Kelvin_per_y; kSZ_return = kSZ_Kelvin_per_t
    else:
        tSZ_return = tSZ_JyBeam_per_y.value; kSZ_return = kSZ_JyBeam_per_t.value        
        
    return tSZ_return, kSZ_return

def get_maps_and_info(instrument,target,real=True):
    """
    Would like to deprecate this.
    """

    fitsfile, wtfile, wtext, wtisrms, tab = get_fits_files(instrument,target,real=True)
    data_map, header = fits.getdata(fitsfile, header=True)
    wt_map = fits.getdata(wtfile,wtext)
    if wtisrms == True:
        wt_map = 1.0/wt_map**2

    image_data, ras, decs, hdr, pixs = get_astro(fitsfile)
    w = wcs.WCS(fitsfile)
    
    return data_map, wt_map, header, ras, decs, pixs, w, tab

#def get_map_info(file):
#
#    image_data, ras, decs, hdr, pixs = get_astro(file)

def get_beamvolume(instrument):

    ### The default is to assume a double Gaussian
    fwhm1,norm1,fwhm2,norm2,fwhm,smfw,freq,FoV = inst_params(instrument)

    sc = 2.0 * np.sqrt(2.0*np.log(2)) # Sigma conversion (from FWHM)
    sig1 = fwhm1/sc                   # Apply the conversion
    sig2 = fwhm2/sc                   # Apply the conversion
    bv1 = 2.0*np.pi * norm1*sig1**2   # Calculate the integral
    bv2 = 2.0*np.pi * norm2*sig2**2   # Calculate the integral
    beamvolume = bv1 + bv2  # In units of FWHM**2 (should be arcsec squared) 

    print 'Using ',beamvolume,' for MUSTANG2.'
#    if instrument == 'MUSTANG2':
#        beamvolume = beamvolume*0.0 + 110.0 # Hopefully this keeps the units...
#print 'Using ',beamvolume,' for MUSTANG2.'
#        import pdb;pdb.set_trace()
    
    return beamvolume

def get_sz_conversion(temp,instrument,beta=0.0,betaz=0.0):
    """
    Returns the tSZ and kSZ conversions to Jy/beam for a given instrument,
    for a given electron temperature, velocity relative to the speed of
    light (beta), and beta along the line of sight (betaz).
    
    Parameters
    __________
    instrument : MUSTANG, BOLOCAM, or NIKA
    target: The target of the cluster
    real: True or False (do you want to analyze/fit real data or
             'virtual' (simulated) data?).import numpy as np

    Returns
    -------
    CtSZ, CkSZ
    """

    bv = get_beamvolume(instrument)
    fwhm1,norm1,fwhm2,norm2,fwhm,smfw,freq,FoV = inst_params(instrument)
 #   conv = tsz.tSZ_conv_single(temp,freq.value)
    bpsr = tsz.intensity_factors(freq.value,bv)
    szcv,szcu=get_sz_values()
    freq_conv = (szcv['planck'] *1.0e9)/(szcv['boltzmann']*szcv['tcmb'])
    temp_conv = 1.0/szcv['m_e_c2']
    conv = tsz.itoh_2004_int(temp*temp_conv,freq*freq_conv)

    print conv,bpsr
    yJyBeam = conv.item(0)*bpsr/(temp*temp_conv) # Conversion from Compton y to Jy/beam

    return yJyBeam

###################################################################
### Under verification below.

def astro_from_hdr(hdr):
    
    xsz = hdr['naxis1']
    ysz = hdr['naxis2']
    xar = np.outer(np.arange(xsz),np.zeros(ysz)+1.0)
    yar = np.outer(np.zeros(xsz)+1.0,np.arange(ysz))
    ####################

    w = wcs.WCS(hdr)
    #import pdb;pdb.set_trace()
    
    xcen = hdr['CRPIX1']
    ycen = hdr['CRPIX2']
    dxa = xar - xcen
    dya = yar - ycen
    ### RA and DEC in degrees:
    if 'CD1_1' in hdr.keys():
        ras = dxa*hdr['CD1_1'] + dya*hdr['CD2_1'] + hdr['CRVAL1']
        decs= dxa*hdr['CD1_2'] + dya*hdr['CD2_2'] + hdr['CRVAL2']
        pixs= abs(hdr['CD1_1'] * hdr['CD2_2'])**0.5 * 3600.0
    if 'PC1_1' in hdr.keys():
        pcmat = w.wcs.get_pc()
        ras = dxa*pcmat[0,0]*hdr['CDELT1'] + \
              dya*pcmat[1,0]*hdr['CDELT2'] + hdr['CRVAL1']
        decs= dxa*pcmat[0,1]*hdr['CDELT1'] + \
              dya*pcmat[1,1]*hdr['CDELT2'] + hdr['CRVAL2']
        pixs= abs(pcmat[0,0]*hdr['CDELT1'] * \
                  pcmat[1,1]*hdr['CDELT2'])**0.5 * 3600.0

    pixs = pixs*u.arcsec
    ### Do I want to make ras and decs Angle objects??
    ras  = ras*u.deg; decs = decs*u.deg 
    
    return ras, decs, pixs

def get_astro(file):

    hdu = fits.open(file)
    hdr = hdu[0].header
    image_data = hdu[0].data

    ras, decs, pixs = astro_from_hdr(hdr)

    return image_data, ras, decs, hdr, pixs

class astrometry:

    def __init__(self,hdr):

        ras,decs,pixs = astro_from_hdr(hdr)
        self.ras = ras
        self.decs= decs
        self.pixs= pixs
        
def get_xfer(inputs):

    if inputs.tabformat == 'ascii':
        tab = np.loadtxt(inputs.tabfile, comments=inputs.tabcomments)
    if inputs.tabformat == 'fits':
        ktab = fits.getdata(inputs.tabfile)
        xtab = fits.getdata(inputs.tabfile,ext=1)
        tab = np.vstack((ktab,xtab))
        
    if inputs.tabdims == '1D':
        if inputs.instrument == "MUSTANG" or inputs.instrument == "MUSTANG2":
            tab = tab.T            # Transpose the table.
#        import pdb;pdb.set_trace()
        if inputs.tabextend==True:
            tdim = tab.shape
#            import pdb;pdb.set_trace()
            pfit = np.polyfit(tab[0,tdim[1]/2:],tab[1,tdim[1]/2:],1)
            addt = np.max(tab[0,:]) * np.array([2.0,4.0,8.0,16.0,32.0])
            extt = np.polyval(pfit,addt)
            foo = np.stack((addt,extt))
            tab = np.concatenate((tab,foo),axis=1)
#            newt = [np.append(tab[0,:],addt),np.append(tab[1,:],extt)]
            
    return tab

############################################################################

def get_conv_factor(instrument):
    
    fwhm1,norm1,fwhm2,norm2,fwhm,smfw,freq,FoV = inst_params(instrument)
    #freq = 90.0 * u.GHz;     instrument='MUSTANG2'
    szcv,szcu = get_sz_values()
    x = szcv["planck"]*(freq.to("Hz")).value / (szcv["boltzmann"]*szcv["tcmb"])
    bv = get_beamvolume(instrument)

    fofx = x * (np.exp(x) + 1.0)/(np.exp(x) - 1.0) - 4.0 # Delta T / T * y
    gofx = fofx * x**4 * np.exp(x) / (np.exp(x) - 1)**2  # Delta I / I * y

    B_nu = 2.0*((const.h*freq**3)/(const.c**2 * u.sr)).decompose()
    B_nu *= 1.0/(np.exp(x) - 1.0)
    JyperSrK = (B_nu/(szcv["tcmb"]*u.K)).to("Jy / (K sr)")
    JyperAsK = JyperSrK.to("Jy / (K arcsec2)")
    
    ### This value assumes true black-body spectrum:
    JyperK_SZ = JyperAsK*bv*gofx/fofx

    ### Radiance per Kelvin:
    I_per_K = 2.0*freq**2 * const.k_B / (const.c**2 * u.sr)
    IpK = I_per_K.to("Hz J s / (K m2 sr)")
    WperSr = IpK.to("W / (Hz K m2 sr)")
    JyperSr = WperSr.to("Jy / (K sr)")
    Jy_per_K_as = JyperSr.to("Jy / (K arcsec2)")

    ###  And this one assumes the RJ law:
    JyperK_RJ = Jy_per_K_as*bv

    return JyperK_RJ

def get_sz_values():
    ########################################################
    ### Astronomical value...
    tcmb = 2.72548*u.K # Kelvin (uncertainty = 0.00057)
    ### Reference:
    ### http://iopscience.iop.org/article/10.1088/0004-637X/707/2/916/meta
    
    ### Standard physical values.
    thom_cross = (spconst.value("Thomson cross section") *u.m**2).to("cm**2")
    m_e_c2 = (const.m_e *const.c**2).to("keV")
    kpctocm = 3.0856776 *10**21
    boltzmann = spconst.value("Boltzmann constant in eV/K")/1000.0 # keV/K  
    planck = spconst.value("Planck constant in eV s")/1000.0 # keV s
    c = const.c
    keVtoJ = (u.keV).to("J") # I think I need this...) 
    Icmb = 2.0 * (boltzmann*tcmb.value)**3 / (planck*c.value)**2
    Icmb *= keVtoJ*u.W *u.m**-2*u.Hz**-1*u.sr**-1 # I_{CMB} in W m^-2 Hz^-1 sr^-1
    JyConv = (u.Jy).to("W * m**-2 Hz**-1")
    Jycmb = Icmb.to("Jy sr**-1")  # I_{CMB} in Jy sr^-1
    MJycmb= Jycmb.to("MJy sr**-1")

    ### The following constants (and conversions) are just the values (in Python):
    sz_cons_values={"thom_cross":thom_cross.value,"m_e_c2":m_e_c2.value,
                    "kpctocm":kpctocm,"boltzmann":boltzmann,
                    "planck":planck,"tcmb":tcmb.value,"c":c.value,}
    ### The following "constants" have units attached (in Python)!
    sz_cons_units={"Icmb":Icmb,"Jycmb":Jycmb,"thom_cross":thom_cross,
                   "m_e_c2":m_e_c2}

    return sz_cons_values, sz_cons_units

def get_SZ_vars(temp=5.0,instrument='MUSTANG2',units='Jy/beam'):

    sz_vars,szcu = get_sz_values()
    tSZ,kSZ = get_sz_bp_conversions(temp,instrument,units=units,array="2",
                                           inter=False,beta=0.0/300.0,
                                           betaz=0.0/300.0,rel=True,quiet=False)
    sz_vars['tSZ'] = tSZ
    sz_vars['kSZ'] = kSZ
    sz_vars['temp']= temp

    return sz_vars

def get_map_vars(cluster_priors,instrument='MUSTANG2'):

    m500   = cluster_priors.M500 * u.M_sun
    z      = cluster_priors.z

    d_ang = get_d_ang(z)
    r500,p500 = R500_P500_from_M500_z(m500,z)
    r500ang   = (r500/d_ang).decompose()
    print r500ang,' in Radians'
    nb_theta_range = 150                   # The number of bins for theta_range
    minpixrad = (1.0*u.arcsec).to('rad')
    tnx       = [minpixrad.value,10.0*r500ang.value]  # In radians
    thetas    = np.logspace(np.log10(tnx[0]),np.log10(tnx[1]), nb_theta_range)
    
    map_vars={"instrument":instrument,"thetas":thetas, "d_ang":d_ang,
              "m500":m500,"r500":r500,"p500":p500,"z":z,
              "racen":cluster_priors.ra.to('deg'),"deccen":cluster_priors.dec.to('deg')}

    return map_vars

def R500_P500_from_M500_z(M500,z):

    dens_crit = cosmo.critical_density(z)
    E   = cosmo.H(z)/cosmo.H(0)
    h70 = (cosmo.H(0) / (70.0*u.km / u.s / u.Mpc))

    
    P500 = (1.65 * 10**-3) * ((E)**(8./3)) * ((
        M500 * h70)/ ((3*10**14) * const.M_sun)
        )**(2./3) * h70**2 * u.keV / u.cm**3
    R500 = (3 * M500/(4*np.pi * 500 * dens_crit))**(1./3)

    return R500, P500

def get_d_ang(z):

    d_ang = cosmo.comoving_distance(z) / (1.0 + z)

    return d_ang

def get_cosmo():

    return cosmo

def get_underlying_vars():

    ### Some cluster-dependent variables:
    rxj1347_priors = rxj1347_priors()
    m500           = rxj1347_priors.M500 * u.M_sun
    z              = rxj1347_priors.z
    #racen  = rxj1347_priors.ra.to('deg')
    #deccen = rxj1347_priors.dec.to('deg')
    ### Some fitting variables:
    beamvolume=120.0 # in arcsec^2
    radminmax = np.array([9.0,4.25*60.0])*(u.arcsec).to('rad')
    nbins     = 6    # It's just a good number...so good, you could call it a perfect number.

    ##############
    bins      = np.logspace(np.log10(radminmax[0]),np.log10(radminmax[1]), nbins) 
    #geom     = [X_shift, Y_shift, Rotation, Ella*, Ellb*, Ellc*, Xi*, Opening Angle]
    geom      = [0,0,0,1,1,1,0,0] # This gives spherical geometry
    map_vars  = get_map_vars(rxj1347_priors, instrument='MUSTANG2')
    alphas    = np.zeros(nbins) #??
    d_ang     = get_d_ang(z)
    #binskpc   = bins * d_ang
    sz_vars,szcu = get_sz_values()
    sz_vars   = get_SZ_vars(temp=rxj1347_priors.Tx)
    Pdl2y     = (szcu['thom_cross']*d_ang/szcu['m_e_c2']).to("cm**3 keV**-1")

    return sz_vars, map_vars, bins, Pdl2y, geom

class rxj1347_priors:
        
    def __init__(self):
        
        ###############################################################################
        ### Prior known values regarding the RXJ1053. Redshift, ra, and dec *MUST* be
        ### known / accurate. M_500 and Tx are useful for creating initial guesses.
        ### Tx is still important if relativistic corrections may be severe.
        
        self.z     = 0.4510                      # Redshift
        self.ra    = Angle('13h47m30.5s')     # Right Ascencion, in hours
        self.dec   = Angle('-11d45m9s')       # Declination, in degrees
        self.M500  = 2.2e15                 # Solar masses
        self.Tx    = 10.8                  # keV
        self.name  = 'rxj1347'
        
        ###  For when the time comes to use the *actual* coordinates for Abell 2146,
        ###  Here they are. Even now, it's useful to calculate the offsets of the centroids
        ###  for the radius of curvature of the shocks.

class zw3146_priors:
        
    def __init__(self):
        
        ###############################################################################
        ### Prior known values regarding the RXJ1053. Redshift, ra, and dec *MUST* be
        ### known / accurate. M_500 and Tx are useful for creating initial guesses.
        ### Tx is still important if relativistic corrections may be severe.
        
        self.z     = 0.291                      # Redshift
        self.ra    = Angle('10h23m39.7087s') # Right Ascencion, in hours
        self.dec   = Angle('+4d11m00.750s')  # Declination, in degrees
        self.M_500 = 6.82e14                 # Solar masses
        self.Tx    = 7.0                     # keV
        self.name  = 'Zw3146'

        ###  For when the time comes to use the *actual* coordinates for Abell 2146,
        ###  Here they are. Even now, it's useful to calculate the offsets of the centroids
        ###  for the radius of curvature of the shocks.

### Copied from gNFW_profiles.py:

def a10_from_m500_z(m500, z,rads):
    """
    INPUTS:
    m500    - A quantity with units of mass
    z       - redshift.
    rads    - A quantity array with units of distance.
    """
    
    r500, p500 = gdi.R500_P500_from_M500_z(m500,z)
    gnfw_prof  = gnfw(r500,p500,rads)

    return gnfw_prof

### Copy this.   
def gnfw(R500, P500, radii, c500= 1.177, p=8.403, a=1.0510, b=5.4905, c=0.3081):

    cosmo = gdi.get_cosmo()
    h70 = (cosmo.H(0) / (70.0*u.km / u.s / u.Mpc))

    P0 = P500 * p * h70**-1.5
    rP = R500 / c500 # C_500 = 1.177
    rf =  (radii/rP).decompose().value # rf must be dimensionless
    result = (P0 / (((rf)**c)*((1 + (rf)**a))**((b - c)/a)))

    return result
    
def compton_y_profile_from_m500_z(M500, z, mycosmo):
    """
    Computes the Compton y profile for an A10 (gNFW) profile, integrated from
    (z = +5*R500) to (z = -5*R500).

    Returns the Compton y profile alongside a radial profile in kpc.
    """

    myprof, radii = pressure_profile_from_m500_z(M500, z, mycosmo, N_R500 = 10.0)

    R500, P500 = R500_P500_from_M500_z(M500, z, mycosmo)
    R_scaled = np.logspace(np.log10(10.0**(-4)), np.log10(5.0), 9000)
    radProjected = R_scaled*R500

    m_e_c2 = (const.m_e *const.c**2).to("keV")
    thom_cross = const.sigma_T

    unitless_profile = (myprof * thom_cross * u.kpc / m_e_c2).decompose()

    inrad = radii.to("kpc"); zvals = radProjected.to("kpc")
    
    yprof = ni.int_profile(inrad.value, unitless_profile.value,zvals.value)

    return yprof, inrad

def Y_cyl(yprof, radii, Rmax, d_ang = 0, z=1.99):

    if d_ang == 0:
        d_a = cd.angular_diameter_distance(z, **cosmo) *u.Mpc
    
    try:
        angle = radii.to("rad")   # Try converting to radians.
        max_angle = Rmax.to("rad")
    except:
        print 'Radii are not given in angular units.'
        print 'Using angular diameter to convert radii to angular units.'
        angle = (radii/d_ang).decompose() * u.rad
        max_angle = (Rmax/d_ang).decompose() * u.rad
    
    goodR  = (angle < max_angle)
    goodprof = yprof[goodR]; goodangle = angle[goodR]
    
    prats = goodprof[:-1] / goodprof[1:]
    arats = goodangle[:-1] / goodangle[1:]
    alpha = np.log(prats) / np.log(arats)

    parint= ((goodangle[1:]/u.rad)**(2.0-alpha) - (goodangle[:-1]/u.rad)**(2.0-alpha) ) * \
            (goodprof[:-1]*(goodangle[:-1]/u.rad)**alpha) / (2.0 - alpha)
    tint  = 2.0*np.pi * np.sum(parint) * u.sr

    Ycyl  = tint.to("arcmin2")

    print 'Ycyl found to be ', Ycyl

    return Ycyl

def Y_sphere(myprof, radii, Rmax, d_ang = 0, z=1.99):
    
    if d_ang == 0:
        d_ang = cd.angular_diameter_distance(z, **cosmo) *u.Mpc

    m_e_c2 = (const.m_e *const.c**2).to("keV")
    thom_cross = const.sigma_T
    unitless_profile = (myprof * thom_cross * u.kpc / m_e_c2).decompose()

    goodR  = (radii < Rmax)
    goodprof = unitless_profile[goodR]; goodradii = radii[goodR]

    prats = goodprof[:-1] / goodprof[1:]
    arats = goodradii[:-1] / goodradii[1:]
    alpha = np.log(prats) / np.log(arats)
     
    parint= ((goodradii[1:]/u.kpc)**(3.0-alpha) - (goodradii[:-1]/u.kpc)**(3.0-alpha) ) * \
            (goodprof[:-1]*(goodradii[:-1]/u.kpc)**alpha) / (3.0 - alpha)
    tint  = 4.0*np.pi * np.sum(parint) * (u.kpc)**2

    Ysphere = tint.to("Mpc2")

    Ysphere_rad = Ysphere / d_ang
    
    print 'Ysphere found to be ', Ysphere
    print 'or alternatively ', Ysphere_rad

    return Ysphere




    
