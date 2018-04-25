import numpy as np                    # A useful module
import astropy.units as u             # U just got imported!
import get_data_info as gdi           # Not much of a joke to make here.
import analytic_integrations as ai    # Well that could be misleading...
import astropy.constants as const     # 
from astropy.coordinates import Angle #
import gNFW_profiles as gp            # Sure seems general purpose
import yafc                           # 


### I do want these to be global.
sz_vars, map_vars, bins, Pdl2y, geom = yafc.get_underlying_vars()
    
def get_prof(ras,decs,pos,posind=0):
  
    alphas    = pos*0.0
    yProf, outalphas = Comptony_profile(pos,posind,bins,sz_vars,map_vars,geom,alphas,
                            fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
                                        finite=False,oldvs=False,fit_cen=False)
    radarr  = radec2rad(ras, decs, map_vars["racen"], map_vars["deccen"], geoparams=[0,0,0,1,1,1,0,0])
    radVals = (radarr.to('rad')).value
    yVals   = np.interp(radVals, map_vars['thetas'],yProf)

    plot_example(radVals,yVals, map_vars['thetas'], yProf)

    return yVals
    
#def get_underlying_vars():
#
#    ### Some cluster-dependent variables:
#    rxj1347_priors = priors()
#    m500   = rxj1347_priors.M500 * u.M_sun
#    z      = rxj1347_priors.z
    #racen  = rxj1347_priors.ra.to('deg')
    #deccen = rxj1347_priors.dec.to('deg')
    ### Some fitting variables:
#    beamvolume=120.0 # in arcsec^2
#    radminmax = np.array([9.0,4.25*60.0])*(u.arcsec).to('rad')
#    nbins     = 6    # It's just a good number...so good, you could call it a perfect number.

    ##############
#    bins      = np.logspace(np.log10(radminmax[0]),np.log10(radminmax[1]), nbins) 
    #geom     = [X_shift, Y_shift, Rotation, Ella*, Ellb*, Ellc*, Xi*, Opening Angle]
#    geom      = [0,0,0,1,1,1,0,0] # This gives spherical geometry
#    map_vars  = gdi.get_map_vars(rxj1347_priors, instrument='MUSTANG2')
#    alphas    = np.zeros(nbins) #??
#    d_ang     = gdi.get_d_ang(z)
    #binskpc   = bins * d_ang
#    sz_vars,szcu = gdi.get_sz_values()
#    sz_vars   = gdi.get_SZ_vars(temp=rxj1347_priors.Tx)
#    Pdl2y     = (szcu['thom_cross']*d_ang/szcu['m_e_c2']).to("cm**3 keV**-1")#
#
#    return sz_vars, map_vars, bins, Pdl2y, geom
    
def example_profile():
    """
    Here, I collect all the necessary inputs. Main goals:
    (1) get pressure profile parameters into a unitless array
    (2) get profile radii as an array, expressed in radians.
    (3) 
    """
    #sz_vars, map_vars, bins, Pdl2y, geom = yafc.get_underlying_vars()

    rads      = bins * map_vars["d_ang"]
    a10pres   = gp.a10_from_m500_z(map_vars["m500"], map_vars["z"], rads)
    uless_p   = (a10pres*Pdl2y).decompose().value   # Unitless array
    alphas    = uless_p*0.0
    pos       = uless_p                             # These to be fed in via MCMC
    posind    = 0
    ras      = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["racen"]
    decs     = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["deccen"]
    yVals   = get_prof(ras,decs,pos,posind=0)

    #yProf, outalphas = Comptony_profile(pos,posind,bins,sz_vars,map_vars,geom,alphas,
    #                        fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
    #                                    finite=False,oldvs=False,fit_cen=False)
    #ras      = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["racen"]
    #decs     = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["deccen"]
    #
    #radarr  = radec2rad(ras, decs, map_vars["racen"], map_vars["deccen"], geoparams=[0,0,0,1,1,1,0,0])
    #radVals = (radarr.to('rad')).value
    #yVals   = np.interp(radVals, map_vars['thetas'],yProf)
    yVals   = get_prof(ras,decs,pos,posind=0)   

#class uvars:#
#
#    def __init__(self):#
#
#        sz_vars, map_vars, bins, Pdl2y, geom = get_underlying_vars()
#
#        self.sz_vars  = sz_vars
#        self.map_vars = map_vars
#        self.bins     = bins
#        self.Pdl2y    = Pdl2y
#        self.geom     = geom
        
class priors:
        
    def __init__(self):
        
        ###############################################################################
        ### Prior known values regarding the RXJ1053. Redshift, ra, and dec *MUST* be
        ### known / accurate. M_500 and Tx are useful for creating initial guesses.
        ### Tx is still important if relativistic corrections may be severe.
        
        self.z=0.4510                      # Redshift
        self.ra = Angle('13h47m30.5s')     # Right Ascencion, in hours
        self.dec= Angle('-11d45m9s')       # Declination, in degrees
        self.M500 = 2.2e15                 # Solar masses
        self.Tx    = 10.8                  # keV
        self.name  = 'rxj1347'
        
        ###  For when the time comes to use the *actual* coordinates for Abell 2146,
        ###  Here they are. Even now, it's useful to calculate the offsets of the centroids
        ###  for the radius of curvature of the shocks.

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
                     finite=False,oldvs=False,fit_cen=False):

    nbins = len(bins)
    posind = 0         # If you have other parameters than the bulk, this may differ.
    if finite == True:
        nbins-=1     # Important correction!!!
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

    return Int_Prof, outalphas

    
def bulk_or_shock_component(pos,posind,bins,sz_vars,map_vars,fit_cen,geom,alphas,
                            fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
                            finite=False,oldvs=False):

    nbins = len(bins)
    posind = 0         # If you have other parameters than the bulk, this may differ.
    if finite == True:
        nbins-=1     # Important correction!!!
    ulesspres = pos[posind:posind+nbins]
    #myalphas  = alphas[posind:posind+nbins]
    myalphas = alphas   # I've updated how I pass alphas; indexing no longer necessary! (16 Nov 2017)
    ulessrad  = bins #.to("rad").value
    posind = posind+nbins
    if fit_cen == True:
        geom[0:2] = pos[posind:posind+2]  # I think this is what I want...
        posind = posind+2

    density_proxy, etemperature, geoparams = ai.prep_SZ_binsky(ulesspres,sz_vars['temp'],geoparams=geom)
    ### Can modify later to allow for X-ray images
    ### That is, I will want to do a loop over SZ images (reduce the number of integrations done),
    ### and then integrate over X-ray emissivity.
    #import pdb;pdb.set_trace()
    
    if fullSZcorr == False:
        #import pdb;pdb.set_trace()
        Int_Pres,outalphas,integrals = ai.integrate_profiles(density_proxy, etemperature, geom,bins,
                 efv.thetas,sz_vars,myalphas,beta=0.0,betaz=None,finint=finite,narm=False,fixalpha=fixalpha,
                 strad=False,array="2",SZtot=False,columnDen=False,Comptony=True)
        yint=ai.ycylfromprof(Int_Pres,efv.thetas,efv.thetamax) #

    ### I think I can just do "for myinst in hk.instruments:"
    #myinst = hk.instruments[i];
    #xymap=dv[myinst].mapping.xymap

    if Comptony == False:
        if fullSZcorr == True:
            IntProf,outalphas,integrals = ai.integrate_profiles(density_proxy, etemperature, geom,bins,
                 efv.thetas,hk,dv,myalphas,beta=0.0,betaz=None,finint=finite,narm=False,fixalpha=fixalpha,
                 strad=False,array="2",SZtot=True,columnDen=False,Comptony=False)
            ### The following is not really correct. As this is under development, I'll leave it for later
            ### to solve.
            yint=ai.ycylfromprof(IntProf,efv.thetas,efv.thetamax) #
            yint=0 # A sure way to give something clearly wrong -> bring myself back here.
            import pdb;pdb.set_trace() # A better way...
        else:
            ### Convert to Column Density (for kSZ)....?
            ConvtoCD= hk.av.szcv["m_e_c2"]/(hk.av.szcv["boltzmann"]*hk.hk_ins.Tx)
            IntProf = Int_Pres * (dv[myinst].tSZ + dv[myinst].kSZ*ConvtoCD)
            integrals = integrals * (dv[myinst].tSZ + dv[myinst].kSZ*ConvtoCD)

        ### Right...I need to have a zero-map to start with. Ughhh.
        #maps[myinst] += ai.general_gridding(xymap,efv.thetas,bins,geom,finite=finite,integrals=integrals,
        #                                    Int_Pres=IntProf,oldvs=oldvs)
        
    return maps,posind,yint,outalphas

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
    
