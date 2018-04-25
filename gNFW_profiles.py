import numpy as np
import scipy.constants as spconst
import astropy.constants as const
import astropy.units as u
import numerical_integration as ni
import get_data_info as gdi

### Copy this from cpp.
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




    
