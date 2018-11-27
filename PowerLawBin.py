import numpy as np                    # A useful module
import astropy.units as u             # U just got imported!
import get_data_info as gdi           # Not much of a joke to make here.
import analytic_integrations as ai    # Well that could be misleading...
import astropy.constants as const     # 
from astropy.coordinates import Angle #
import gNFW_profiles as gp            # Sure seems general purpose

### I do want these to be global.
sz_vars, map_vars, bins, Pdl2y, geom = gdi.get_underlying_vars()

ra0  = map_vars["racen"].to('rad').value
dec0 = map_vars["deccen"].to('rad').value

def prof_from_rads(pos,bins,posind=0):

    alphas    = pos*0.0
    yProf, outalphas = Comptony_profile(pos,posind,bins,sz_vars,map_vars,geom,alphas,
                            fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
                                        finite=False,oldvs=False,fit_cen=False)
    return yProf


def get_prof(ras,decs,pos,posind=0):
  
    alphas    = pos*0.0
    yProf, outalphas = Comptony_profile(pos,posind,bins,sz_vars,map_vars,geom,alphas,
                            fixalpha=False,fullSZcorr=False,SZtot=False,columnDen=False,Comptony=True,
                                        finite=False,oldvs=False,fit_cen=False)
    #radarr  = radec2rad(ras, decs, map_vars["racen"], map_vars["deccen"], geoparams=[0,0,0,1,1,1,0,0])
    #radVals = (radarr.to('rad')).value

    radVals = radec2rad(ras, decs, ra0, dec0, geoparams=[0,0,0,1,1,1,0,0])
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
    a10pres   = gp.a10_from_m500_z(map_vars["m500"], map_vars["z"], rads)
    uless_p   = (a10pres*Pdl2y).decompose().value   # Unitless array
    alphas    = uless_p*0.0
    pos       = uless_p                             # These to be fed in via MCMC
    posind    = 0
    ras      = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["racen"]
    decs     = ((np.random.rand(10000)*0.2 -0.1) * u.deg) + map_vars["deccen"]
    ras      = ras.to('rad').value
    decs     = decs.to('rad').value
    
    yVals   = get_prof(ras,decs,pos,posind=0)   

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
    
