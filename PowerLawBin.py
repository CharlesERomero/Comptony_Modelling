import numpy as np                    # A useful module
import astropy.units as u             # U just got imported!
import get_data_info as gdi           # Not much of a joke to make here.
import analytic_integrations as ai    # Well that could be misleading...
#from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
import numerical_integration as ni    #
import scipy.special as scs

#import astropy.constants as const     # 
#from astropy.coordinates import Angle #

### I do want these to be global.
##################################
#def get_base_vars(cluster='Zw3146'):

gcluster = 'Zw3146'
#gcluster = 'HSC_2'
#gcluster = 'MOO_0105'
#gcluster = 'MOO_0135'
#gcluster = 'MOO_1014'
#gcluster = 'MOO_1046'
#gcluster = 'MOO_1059'
#gcluster = 'MOO_1108'
#gcluster = 'MOO_1110'
#gcluster = 'MOO_1142'
#############################
#gcluster = 'MACS0717'
#gcluster = 'MACS1149'
### or... 'HSC_2'

#mygeom      = [0,0,0.8,1,0.8,0.8944272,0,0]
#mygeom      = [-1.96,-0.26,0.8,1,0.8,0.8944272,0,0]  # If elliptical, SZ2D
sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(gcluster)
tM2c,  kM2c    = gdi.get_sz_bp_conversions(sz_vars['temp'],'MUSTANG2',units='Kelvin', inter=False,
                                           beta=0.0/300.0,betaz=0.0/300.0,rel=True,quiet=False,
                                           cluster=gcluster,RJ=True)
tA090c, kA090c = gdi.get_sz_bp_conversions(sz_vars['temp'],'ACT90',units='Kelvin', inter=False,
                                           beta=0.0/300.0,betaz=0.0/300.0,rel=True,quiet=False,
                                           cluster=gcluster)
tA150c, kA150c = gdi.get_sz_bp_conversions(sz_vars['temp'],'ACT150',units='Kelvin', inter=False,
                                           beta=0.0/300.0,betaz=0.0/300.0,rel=True,quiet=False,
                                           cluster=gcluster)

ra0  = map_vars["racen"].to('rad').value
dec0 = map_vars["deccen"].to('rad').value

    #return sz_vars, map_vars, bins, Pdl2y, geom, tM2c, kM2c, tA090c, kA090c, tA150c, kA150c

def get_conv_factors(instrument='MUSTANG2'):

    if instrument == 'MUSTANG2':
        tc,kc = tM2c,  kM2c
    if instrument == 'ACT90':
        tc,kc = tA090c,  kA090c
    if instrument == 'ACT150':
        tc,kc = tA150c,  kA150c

    return tc,kc

def get_linear_bins():

    ppbins  = np.arange(2.0,200.0,4.0) # The spacing in arcseconds
    ppbins *= np.pi / (3600.0*180.0)   # Now in radians

    return ppbins

def calc_ySph(pos,ppbins):

    print('hi')
    import pdb;pdb.set_trace()

def a10_pres_from_rads(ppbins):

    rads      = ppbins * map_vars["d_ang"]
    a10pres   = gdi.a10_from_m500_z(map_vars["m500"], map_vars["z"], rads)
    uless_p   = (a10pres*Pdl2y).decompose().value   # Unitless array

    return uless_p

def a10_prof_from_rads(ppbins):

    """
    ppbins needs to be in radians.; the code will transform it to physical units. These bins
    serve as the radii for the pressure profile normalizations, between which the pressure
    profile is logarithmically interpolated.
    """
    
    uless_p   = a10_pres_from_rads(ppbins)   # Unitless array
    alphas    = uless_p*0.0
    pos       = uless_p                             # These to be fed in via MCMC
    posind    = 0

    #import pdb;pdb.set_trace()

    yProf = prof_from_pars_rads(pos,ppbins,posind)

    return yProf
    
def prof_from_pars_rads(pos,ppbins,posind=0,retall=False,model='NP',cluster='Default',ySph=False,
                        geom=[0,0,0,1,1,1,0,0]):
    """
    ppbins needs to be in radians; the code will transform it to physical units. These bins
    serve as the radii for the pressure profile normalizations, between which the pressure
    profile is logarithmically interpolated.

    The Comton y profile (yProf) will be calculated for the angular radii given by
    map_vars['thetas'], which are expressed in radians.
    """

    if cluster != gcluster:
        raise Exception('The cluster you input is NOT the same as the cluster PowerLawBin is using!')

    if model == 'NP':
        alphas    = pos*0.0
        yProf, outalphas,yint = Comptony_profile(pos,posind,ppbins,sz_vars,map_vars,geom,alphas,
                            fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
                                                 finite=False,oldvs=False,fit_cen=False,ySph=ySph)
        if ySph:
            # Do this calculation with "unitless" variables.
            ### I should totally be able to speed up ai.log_profile...
            
            pprof,alphas   = ai.log_profile(pos[:-1],list(ppbins),map_vars['thetas']) # Last pos is mn lvl
            palphas, pnorm = ai.ycyl_prep(pprof,map_vars['thetas'])
            yint, newr500  = ysph_simul(map_vars['thetas'],pprof,palphas,geom)
            
    if model == 'GNFW':
        ### I've set it up so that ppbins is no longer the radii associated with pressure normalizations
        ### when running a non-parametric (NP) model. 
        
        radii    = (map_vars['thetas']* map_vars['d_ang']).to('kpc')
        #radii    = (ppbins * map_vars['d_ang']).to('kpc')     # This will be in kpc
        #import pdb;pdb.set_trace()
        myup     = np.array([1.177,8.403,5.4905,0.3081,1.0510])
        mypos    = pos[:-1]
        if len(mypos) < len(myup):
            #myup[1:1+len(pos)]=pos
            #myup[1:1+len(mypos)]=mypos
            myup[0:len(mypos)]=mypos
        else:
            myup = mypos    # If len(pos) > 5, that's OK...we won't use those!
           
        A10all = 1.0510 # Alpha value in A10, all clusters
        A10cc  = 1.2223 # Alpha value in A10, cool-core clusters
        A10dis = 1.4063 # Alpha value in A10, disturbed clusters

        R500 = map_vars['r500'].to('kpc')
        P500 = map_vars['p500'].to('keV cm**-3')
        r500 = (R500 / map_vars['d_ang']).decompose().value
        
        pprof    = gdi.gnfw(R500, P500, radii, c500=myup[0], p=myup[1], a=myup[4], b=myup[2], c=myup[3])
        unitless_profile = (pprof * Pdl2y).decompose().value
        #inrad = radii.to("kpc"); zvals = radProjected.to("kpc")
        
        #print myup
        #import pdb;pdb.set_trace()

        #yProf = ni.int_profile(radii.value, unitless_profile,radii.value)
        yProf = ni.int_profile(map_vars['thetas'], unitless_profile,map_vars['thetas'])
        outalphas = unitless_profile*0.0+2.0
        integrals = yProf
        #yint = ni.Ycyl_from_yProf(yProf,ppbins,r500)
        if ySph:
            palphas, pnorm = ai.ycyl_prep(unitless_profile,map_vars['thetas'])
            yint, newr500  = ysph_simul(map_vars['thetas'],unitless_profile,palphas,geom)
        else:
            yint ,newr500=Y_SZ_via_scaling(yProf,map_vars["thetas"],map_vars['r500'],map_vars['d_ang'],geom) # As of Aug. 31, 2018
        #import pdb;pdb.set_trace()

    if model == 'Beta':

        ### All values in pos are unitless within Python. But, we'll make pos[1] have units of kpc
        radii    = (map_vars['thetas']* map_vars['d_ang']).to('kpc').value 
        pres     = pos[0]*(1.0+(radii/pos[1])**2)**(-1.5*pos[2])        ### Beta model
        #scaling  = scs.gamma(1.5*pos[2] - 0.5)/scs.gamma(1.5*pos[2]) * pos[1] * pos[0]
        scaling  = scs.gamma(1.5*pos[2]-0.5)/scs.gamma(1.5*pos[2])*(pos[1]/map_vars['d_ang'].to('kpc').value)
        scaling *= pos[0] * np.sqrt(np.pi)
        yProf    = scaling * Pdl2y.value *  (1.0+(radii/pos[1])**2)**(0.5-1.5*pos[2])
        outalphas = pres*0.0+2.0
        #yint ,newr500=Y_SZ_via_scaling(yProf,map_vars["thetas"],map_vars['r500'],map_vars['d_ang']) # As of Aug. 31, 2018
        R500 = map_vars['r500'].to('kpc')
        #P500 = map_vars['p500'].to('keV cm**-3')
        r500 = (R500 / map_vars['d_ang']).decompose().value
        integrals = yProf
        yint = ni.Ycyl_from_yProf(yProf,ppbins,r500)
        
    #import pdb;pdb.set_trace()

    if retall == True: return yProf,outalphas,yint
    return yProf


def resample_prof(yProf,edges,inst='MUSTANG2',slopes=None):

    """
    Slopes are defined relative to arcseconds (radius)
    """
    
    nrings    = len(edges)-1
    if slopes is None:
        slopes = np.zeros((nrings))
        
        
    edgy_rads = edges * np.pi / (180.0*3600.0)
    newProf   = np.zeros(nrings)
    for i in range(nrings):
        gRads1 = (map_vars['thetas'] >=  edgy_rads[i]) # Good radii (condition 1)
        gRads2 = (map_vars['thetas'] < edgy_rads[i+1]) # Good radii (condition 2)
        gRads  = [gRad1 and gRad2 for gRad1,gRad2 in zip(gRads1,gRads2)]
        myrads = map_vars['thetas'][gRads]
        myvals = yProf[gRads]
        a      = 1.0 - slopes[i]*(myrads-edgy_rads[i]) * (3600*180/np.pi)
        ata    = np.sum(a**2 * myrads)     # myrads acts as the N^-1 weighting mechanism
        atd    = np.sum(a*myvals * myrads) # myrads acts as the N^-1 weighting mechanism
        wtavg  = atd/ata
        #wtavg  = np.sum(myvals*myrads)/np.sum(myrads)  # Larger radii will naively contribute more.
        #nom    = np.sum(myvals*(1.0 + slopes*myrads)*myrads)
        #den    = np.sum((1.0 + slopes*myrads)*myrads)
        newProf[i] = wtavg

    if inst == 'MUSTANG2': myconv = tM2c
    if inst == 'ACT90':    myconv = tA090C
    if inst == 'ACT150':   myconv = tA150C

    newProf *= myconv
    
    return newProf
        
def get_mapprof(ras,decs,pos,posind=0,geom=[0,0,0,1,1,1,0,0]):
  
    alphas    = pos*0.0
    yProf, outalphas = Comptony_profile(pos,posind,bins,sz_vars,map_vars,geom,alphas,
                            fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
                                        finite=False,oldvs=False,fit_cen=False,ySph=False)
    #radarr  = radec2rad(ras, decs, map_vars["racen"], map_vars["deccen"], geoparams=[0,0,0,1,1,1,0,0])
    #radVals = (radarr.to('rad')).value

    #radVals = radec2rad(ras, decs, ra0, dec0, geoparams=[0,0,0,1,1,1,0,0])
    radVals = radec2rad(ras, decs, ra0, dec0, geoparams=geom)
    #import pdb;pdb.set_trace()
    yVals   = np.interp(radVals, map_vars['thetas'],yProf)

    #plot_example(radVals,yVals, map_vars['thetas'], yProf)

    return yVals
        
def example_profile():
    """
    Here, I collect all the necessary inputs. Main goals:
    (1) get pressure profile parameters into a unitless array
    (2) get profile radii as an array, expressed in radians.
    (3) 
    """
    rads      = bins * map_vars["d_ang"]
    a10pres   = gdi.a10_from_m500_z(map_vars["m500"], map_vars["z"], rads)
    uless_p   = (a10pres*Pdl2y).decompose().value   # Unitless array
    alphas    = uless_p*0.0
    pos       = uless_p                             # These to be fed in via MCMC
    posind    = 0
    ras      = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["racen"]
    decs     = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["deccen"]
    ras      = ras.to('rad').value
    decs     = decs.to('rad').value
    
    yVals   = get_mapprof(ras,decs,pos,posind=0)   

def radec2rad(ras, decs, racen, deccen, geoparams=[0,0,0,1,1,1,0,0]):

    x,y = get_xymap(ras, decs, racen, deccen)
    x,y = rot_trans_grid(x,y,geoparams[0],geoparams[1],geoparams[2])
    x,y = get_ell_rads(x,y,geoparams[3],geoparams[4])
    radmap = np.sqrt(x**2 + y**2)

    return radmap

def get_xymap(ras, decs, racen, deccen):

    x = ras - racen
    y = decs - deccen

    return x,y

def rot_trans_grid(x,y,xs,ys,rot_rad):

    xnew = (x - xs)*np.cos(rot_rad) + (y - ys)*np.sin(rot_rad)
    ynew = (y - ys)*np.cos(rot_rad) - (x - xs)*np.sin(rot_rad)

    return xnew,ynew

def get_ell_rads(x,y,ella,ellb):

    xnew = x/ella ; ynew = y/ellb

    return xnew, ynew
    

def Comptony_profile(pos,posind,bins,sz_vars,map_vars,geom,alphas,
                     fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
                     finite=False,oldvs=False,fit_cen=False,ySph=False):

    nbins = len(bins)
    posind = 0         # If you have other parameters than the bulk, this may differ.
    if finite == True:
        nbins-=1     # Important correction!!!
    #import pdb;pdb.set_trace()
    ulesspres = pos[posind:posind+nbins]
    #myalphas  = alphas[posind:posind+nbins]
    myalphas = alphas   # I've updated how I pass alphas; indexing no longer necessary! (16 Nov 2017)
    ulessrad  = bins #.to("rad").value
    posind = posind+nbins
    if fit_cen == True:
        geom[0:2] = pos[posind:posind+2]  # I think this is what I want...
        posind = posind+2

    density_proxy, etemperature, geoparams = ai.prep_SZ_binsky(ulesspres,sz_vars['temp'],geoparams=geom)
    
    if fullSZcorr == False:
        #import pdb;pdb.set_trace()
        Int_Prof,outalphas,integrals = ai.integrate_profiles(density_proxy, etemperature, geom,bins,
                 map_vars["thetas"],sz_vars,myalphas,beta=0.0,betaz=None,finint=finite,narm=False,fixalpha=fixalpha,
                                                strad=False,array="2",SZtot=False,columnDen=False,Comptony=Comptony)
        #yint=ai.ycylfromprof(Int_Pres,efv.thetas,efv.thetamax) #
        yint = 0
        if ySph == False:
            yint ,newr500=Y_SZ_via_scaling(Int_Prof,map_vars["thetas"],map_vars['r500'],map_vars['d_ang'],geom) # As of Aug. 31, 2018

    return Int_Prof, outalphas, yint

def plot_example(radVals,yVals, thetas, yProf):

    import matplotlib.pyplot as plt
    plt.figure(2,figsize=(20,12));    plt.clf()
    #plt.axvline(rin,color=axcol, linestyle ="dashed")
    #plt.axvline(rout,color=axcol, linestyle ="dashed")
    plt.plot(radVals,yVals,"o",label = "Interpollated Pressure")
    plt.plot(thetas, yProf)

    runits = 'Radians'
    
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Radius ("+runits+")")
    plt.ylabel("Compton y")
    plt.title("RXJ 1347")
    plt.grid()

    filename = "example_plot.png"
    #savedir  = "/home/romero/Python/StandAlone/Comptony_Modelling/"
    #fullpath = savedir+filename
    plt.savefig(filename)
    #plt.close()
    
def Y_SZ_via_scaling(yProf,rProf,r500,d_ang,mygeom):
    """
    Int_Pres     - an integrated profile, i.e. in Compton y that matches theta_range
    theta_range  - the radial profile on the sky (in radians)
    theta_max    - the maximum (e.g. R500)
    hofz         - little h (as a function of redshift)
    ang_dist     - given as a value, scaled to Mpc.
    nsamp        - number of trial points
    
    I'm adopting equations 25-27 in Arnaud+ 2010, which makes use of Y_SZ, or Y_cyl and the
    Universal Pressure Profile (UPP). I tried to find just a straight empirical Y_cyl(R500)-M_500,
    but that doesn't seem to exist?!? 

    I might move to calculating Y_sph at some point (Mar 11, 2019).
    """
    alpha,norm  = ai.ycyl_prep(yProf,rProf)
    #r_max       = (r500/d_ang).decompose().value   # in radians then.
    #yinteg, root = ycyl_simul_r500(rProf,yProf,alpha,r_max,map_vars['d_ang'])
    yinteg, root = ycyl_simul_v2(rProf,yProf,alpha,mygeom)

    return yinteg, root

def ycyl_simul_r500(rads,yProf,alpha,maxrad,d_a,geom,r_thresh=3e-2):
    """
    This is a BAD, OUTDATED VERSION. TO REMOVE BY APRIL 2019?
    Remember, theta_range is in radians
    """

    fgeo          = geom[3]*geom[4]*geom[5]
    ### In accordance with how Arnaud+ 2010 defines these terms...
    h70      = (cosmo.H(0) / (70.0*u.km / u.s / u.Mpc))
    rho_crit = cosmo.critical_density(map_vars["z"])
    hofz     = cosmo.H(map_vars["z"])/cosmo.H(0)                    

    Ycyl          = 0
    guess_r500    = maxrad/5.0
    goodind       = (rads < guess_r500)
    badind        = (rads >= guess_r500)
    myrads        = rads[goodind]
    #myrads[0]     = 0.0
    if alpha[0]  <= -2: alpha[0]=-1.9
    badalp        = (alpha == -2)
    alpha[badalp] = -2.01 # Va fanculo.
    galphas       = alpha[goodind]
    rolledrad     = np.roll(myrads,-1)
    intupper      = rolledrad**2 * (rolledrad/myrads)**(galphas) #* myrads
    intlower      = myrads**2
    intlower[0]   = 0.0
    integrand     = intupper - intlower
    mynorms       = yProf[goodind]
    Yshell        = 2.0*np.pi*mynorms[:-1]*integrand[:-1]/(galphas[:-1]+2.0)*fgeo
    Ycyl          = np.sum(Yshell)

    #r500          = r500_from_y500(Ycyl,hofz,d_a,rho_crit,h70)
    r500,m500,p500,msys = gdi.rMP500_from_y500(Ycyl,map_vars,ySZ=True)
    newrads       = rads*1.0
    newalpha      = alpha*1.0
    newnorms      = yProf*1.0

        
    while r500 > guess_r500:
        newrads       = np.hstack((myrads[-1],newrads[badind]))
        newalpha      = np.hstack((galphas[-1],newalpha[badind]))
        newnorms      = np.hstack((mynorms[-1],newnorms[badind]))
        goodind       = (newrads < r500)
        badind        = (newrads >= r500)
        myrads        = newrads[goodind]
        galphas       = newalpha[goodind]
        mynorms       = newnorms[goodind]
        rolledrad     = np.roll(myrads,-1)
        intupper      = rolledrad**2 * (rolledrad/myrads)**(galphas) #* myrads
        intlower      = myrads**2
        integrand     = intupper - intlower
        Yshell        = 2.0*np.pi*mynorms[:-1]*integrand[:-1]/(galphas[:-1]+2.0)
        Ycyl         += np.sum(Yshell)
        guess_r500    = r500
        #r500          = r500_from_y500(Ycyl,hofz,d_a,rho_crit,h70)
        r500,m500,p500,msys = gdi.rMP500_from_y500(Ycyl,map_vars,ySZ=True)
    
    delta_r       = r500 - myrads[-1]
    addY          = (r500**2 * (r500/myrads[-1])**(galphas[-1]) - myrads[-1]**2)/(galphas[-1]+2.0)
    propYcyl      = Ycyl + mynorms[-1] * addY * np.pi * 2.0
    finc          = propYcyl/Ycyl
    fadjust       = finc**(1.0/5.0)  ### This had ought to be *very* close to 1.0
    r500         *= fadjust
    
    newrads       = np.hstack((myrads[-1],r500))
    newnorms      = mynorms[-1]

    intupper      = newrads[1]**2 * (newrads[1]/newrads[0])**(galphas[-1])
    intlower      = newrads[0]**2
    integrand     = intupper - intlower

    Yshell        = np.pi*newnorms*integrand/(galphas[-1]+2.0)
    Ycyl         += np.sum(Yshell)
    guess_r500    = r500
    #r500          = r500_from_y500(Ycyl,hofz,d_a,rho_crit,h70)
    r500,m500,p500,msys = gdi.rMP500_from_y500(Ycyl,map_vars,ySZ=True)

    mydiff        = np.abs(guess_r500 - r500)

    #import pdb;pdb.set_trace()
    
    if mydiff > r_thresh*r500: 
        print 'Bad profile: ', guess_r500, r500, maxrad
        return Ycyl,r500
    else:
        return Ycyl,r500

def ycyl_simul_v2(rads,yProf,alpha,mygeom):
    """
    Remember, theta_range is in radians
    """

    ### In accordance with how Arnaud+ 2010 defines these terms...
    h70      = (cosmo.H(0) / (70.0*u.km / u.s / u.Mpc))
    rho_crit = cosmo.critical_density(map_vars["z"])
    hofz     = cosmo.H(map_vars["z"])/cosmo.H(0)                    

    fgeo          = mygeom[3]*mygeom[4] # Scale by ellipsoidal radii scalings.
    Ycyl          = 0
    if alpha[0]  <= -2: alpha[0]=-1.9
    badalp        = (alpha == -2)
    alpha[badalp] = -2.01 # Va fanculo.
    rolledrad     = np.roll(rads,-1)
    intupper      = rolledrad**2 * (rolledrad/rads)**(alpha) #* myrads
    intlower      = rads**2
    intlower[0]   = 0.0
    integrand     = intupper - intlower
    Yshell        = 2.0*np.pi*yProf[:-1]*integrand[:-1]/(alpha[:-1]+2.0)*fgeo
    Ycyl          = np.cumsum(Yshell)
    #import pdb;pdb.set_trace()
    Yref          = map_vars["y500s"][1:]
    mydiff        = Yref - Ycyl
    #r500,m500,p500 = gdi.rMP500_from_y500(Ycyl,map_vars,ySZ=True)

    #mydiff = rads[1:] - r500
    #absdif = np.abs(mydiff)
    posdiffs = (mydiff > 0)
    turnover = mydiff[posdiffs]
    #import pdb;pdb.set_trace()
    bestr = rads[51]
    bestY = Ycyl[50]
    bisca = 0
    if len(turnover) > 1:
        besti  = np.where(mydiff == np.min(turnover))
        bisca  = np.asscalar(besti[0])
    if bisca < map_vars["nrbins"]-3 and bisca > 10: 
        myinds = bisca + np.asarray([-2,-1,0,1,2],dtype='int')
        #myinds = np.intersect1d(naind,
        myrs   = rads[myinds+1]
        myYs   = Ycyl[myinds]
        myds   = mydiff[myinds]
        myp2   = np.polyfit(myrs,myds,2)
        myY2   = np.polyfit(myrs,myYs,2)
        myroot = np.roots(myp2)
        rdiff  = np.abs(myroot - myrs[2])

        bestr  = myroot[0] if rdiff[0] < rdiff[1] else myroot[1]
        Y2fxn  = np.poly1d(myY2)
        bestY  = Y2fxn(bestr)

        if bestY > 3.0e-8:
            print bestr, bestY, np.max(rads)
            stupid = np.random.normal(0,1)
            if stupid > 5: import pdb;pdb.set_trace()
            
        
    #import pdb;pdb.set_trace()
    #bestr  = r500[besti]
    #bestY  = Ycyl[besti]

    return bestY,bestr

###########################################################################

def ysph_simul(rads,pProf,alpha,mygeom):
    """
    To be developped.
    """

    ### In accordance with how Arnaud+ 2010 defines these terms...
    h70      = (cosmo.H(0) / (70.0*u.km / u.s / u.Mpc))
    rho_crit = cosmo.critical_density(map_vars["z"])
    hofz     = cosmo.H(map_vars["z"])/cosmo.H(0)                    

    fgeo          = mygeom[3]*mygeom[4]*mygeom[5] # Scale by ellipsoidal radii scalings.
    Ysph          = 0
    if alpha[0]  <= -3: alpha[0]=-2.9
    badalp        = (alpha == -3)
    alpha[badalp] = -3.01 # Va fanculo.
    rolledrad     = np.roll(rads,-1)
    intupper      = rolledrad**3 * (rolledrad/rads)**(alpha) #* myrads
    intlower      = rads**3
    intlower[0]   = 0.0
    integrand     = intupper - intlower
    Yshell        = 4.0*np.pi*pProf[:-1]*integrand[:-1]/(alpha[:-1]+3.0) 
    Ysph          = np.cumsum(Yshell) *fgeo
    #import pdb;pdb.set_trace()
    #Yref          = map_vars["ySph500s"][1:]
    Yref          = map_vars["y500s"][1:]
    mydiff        = Yref - Ysph
    #r500,m500,p500 = gdi.rMP500_from_y500(Ysph,map_vars,ySZ=True)

    #mydiff = rads[1:] - r500
    #absdif = np.abs(mydiff)
    posdiffs = (mydiff > 0)
    turnover = mydiff[posdiffs]
    #import pdb;pdb.set_trace()
    bestr = rads[51]
    bestY = Ysph[50]
    bisca = 0
    if len(turnover) > 1:
        besti  = np.where(mydiff == np.min(turnover))
        bisca  = np.asscalar(besti[0])
    if bisca < map_vars["nrbins"]-3 and bisca > 10: 
        myinds = bisca + np.asarray([-2,-1,0,1,2],dtype='int')
        #myinds = np.intersect1d(naind,
        myrs   = rads[myinds+1]
        myYs   = Ysph[myinds]
        myds   = mydiff[myinds]
        myp2   = np.polyfit(myrs,myds,2)
        myY2   = np.polyfit(myrs,myYs,2)
        myroot = np.roots(myp2)
        rdiff  = np.abs(myroot - myrs[2])

        bestr  = myroot[0] if rdiff[0] < rdiff[1] else myroot[1]
        Y2fxn  = np.poly1d(myY2)
        bestY  = Y2fxn(bestr)

        if bestY > 3.0e-8:
            print bestr, bestY, np.max(rads)
            stupid = np.random.normal(0,1)
            if stupid > 5: import pdb;pdb.set_trace()
            
        
    #import pdb;pdb.set_trace()
    #bestr  = r500[besti]
    #bestY  = Ysph[besti]

    return bestY,bestr
   
