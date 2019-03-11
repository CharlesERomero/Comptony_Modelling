import numpy as np
from astropy.io import fits
from astropy import coordinates
import scipy.ndimage
from astropy import wcs
from astropy.coordinates import Angle
from astropy.modeling import models, fitting
import warnings
import astropy.units as u             # U just got imported!
import matplotlib.pyplot as plt
import scipy.optimize as opt
import get_data_info as gdi

def get_centroid(cluster):

    if cluster == 'Zw3146':
        ra    = Angle('10h23m39.3336s') # From latest fits (March 1st, 2019)
        dec   = Angle('4d11m14.1248s')  # From latest fits (March 1st, 2019)

    return ra,dec

def get_xymap_hdr(cluster,img,hdr,rwcs=False):

    ras, decs, pixs = astro_from_hdr(hdr)
    ra0, dec0 = get_centroid(cluster)
    w         = wcs.WCS(hdr)
    mypix     = w.wcs_world2pix(ra0.to('degree'),dec0,1)
    print 'Center pixels are: ',mypix
    #import pdb;pdb.set_trace()
    xymap     = get_xymap(img,pixs,mypix[0],mypix[1],oned=False)

    if rwcs:
        return xymap,w,pixs
    else:
        return xymap

def outdir_by_cluster(cluster):

    if cluster == 'Zw3146':
        outdir='/home/data/MUSTANG2/AGBT17_Products/Zw3146/'

    return outdir
        
def find_files(cluster,reduc='IDL',myiter=5):

    if cluster == 'Zw3146':
        if reduc == 'IDL':
            mydir = '/home/romero/Results_Python/MUSTANG2/zw3146/'
            myfile= 'MUSTANG2_Real_Long_Run_MUSTANG2_Models_and_Residuals.fits'
            mext  = 1; snrext=2
        if reduc == 'Minkasi':
            mydir = '/home/romero/Results_Python/Rings/Zw3146/Iter/'
            myfile= 'M2_Zw3146_NPNP-Mar6_v2-Iter'+repr(myiter)+'Iter1_Residual.fits'
            mext  = 2; snrext=3


    return mydir+myfile, mext,snrext
            
def get_maps_file(file,mext,snrext):

    hdu    = fits.open(file)
    hdr    = hdu[mext].header
    img    = hdu[mext].data
    snrhdr = hdu[snrext].header
    snrimg = hdu[snrext].data

    return img,hdr,snrimg,snrhdr

def get_maps_clus(cluster,reduc='IDL',myiter=5):

    file,mext,snrext = find_files(cluster,reduc=reduc,myiter=myiter)
    img,hdr,snrimg,snrhdr = get_maps_file(file,mext,snrext)
    return img,hdr,snrimg,snrhdr

def play_with_resids(cluster='Zw3146',reduc='Minkasi',myiter=5,filename='GaussResid',
                     zoom=6.0):

    img,hdr,snrimg,snrhdr = get_maps_clus(cluster,reduc=reduc,myiter=myiter)
    xymap,w,pixs = get_xymap_hdr(cluster,img,hdr,rwcs=True)
    x,y = xymap
    r = np.sqrt(x*x +y*y)

    smfw = 10.0 * u.arcsec
    sigma = (smfw/pixs).value/(2.0*np.sqrt(2.0*np.log(2.0)))
    img = scipy.ndimage.filters.gaussian_filter(img,sigma)

    #gi = (r.flatten() < 60.0) # All pixels within an arcminute of the center
    gi = (r < 60.0) # All pixels within an arcminute of the center
    snrmed = np.median(snrimg[gi])
    imgmed = np.median(img[gi])
    
    myimg = img.copy()
    mysnr = snrimg.copy()

    if reduc == 'Minkasi':
        myimg -= imgmed
        mysnr -= snrmed

        estra  =  Angle('10h23m38.8266s') # From latest fits (March 1st, 2019)
        estdec =  Angle('4d11m03.755s')  # From latest fits (March 1st, 2019)
    else:
        estra  =  Angle('10h23m37.6590s') # From latest fits (March 1st, 2019)
        estdec =  Angle('4d11m12.557s')  # From latest fits (March 1st, 2019)
        
    x0,y0 = w.wcs_world2pix(estra.to('degree'),estdec,1) # Relative to map...as Python thinks
    x0 += np.min(x)/pixs.value;    y0 += np.min(y)/pixs.value
    xstd = 3.0;         ystd = 3.0          # Arcseconds
    theta = 1.0                             # Radians. How should I know?

    myx   = x - x0; myy = y - y0
    myr   =  np.sqrt(myx*myx +myy*myy)
    besti = (myr < 30.0)
    jfc   = (myr > 35.0)

    fimg  = img.copy()
    fimg[besti] = 0.0
    fimg[jfc]   = 0.0
    myannulus   = (fimg > 0)
    anmed       = np.median(fimg[myannulus])

    img    -= anmed
    myamp = np.max(img[besti])

    #import pdb;pdb.set_trace()


    mnlvl = 0.000001

    vmin = np.min(img[gi])
    vmax = np.max(img[gi])
    z = img.copy()
    zin = z[besti]
    fit_p = fitting.LevMarLSQFitter()

    initial_guess = (myamp,x0,y0,xstd,ystd,theta,mnlvl)
    popt, pcov = opt.curve_fit(my2dGauss, (x[besti], y[besti]), zin, p0=initial_guess)

    ############################################################################
    #g2dbounds = {'x_mean':[x0-10.0,x0+10.0],'y_mean':[y0-10.0,y0+10.0],
    #             'x_stddev':[1.0,20.0],'y_stddev':[1.0,20.0]}
    #p_init = models.Gaussian2D(amplitude=myamp,x_mean=x0,y_mean=y0,
    #                           x_stddev=xstd,y_stddev=ystd,theta=theta,
    #                           bounds=g2dbounds)
    #with warnings.catch_warnings():
    #    # Ignore model linearity warning from the fitter
    #    warnings.simplefilter('ignore')
    #    p = fit_p(p_init, x[besti], y[besti], z[besti])
    ############################################################################

    xsz,ysz = img.shape
    xl,xu   = 0,xsz
    yl,yu   = 0,ysz
    if zoom > 1:
        myf     = 0.5 - 1.0/(2.0*zoom)
        filename=filename+'_zoom'
        xl,xu = int(xsz*myf),int(xsz*(1-myf))
        yl,yu = int(ysz*myf),int(ysz*(1-myf))

    p_fit = my2dGauss((x, y),popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]) 
    z_fit = p_fit.reshape(x.shape)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(z[xl:xu,yl:yu], origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(z_fit[xl:xu,yl:yu], origin='lower', interpolation='nearest', vmin=vmin,
               vmax=vmax)
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(z[xl:xu,yl:yu] - z_fit[xl:xu,yl:yu], origin='lower', interpolation='nearest', vmin=vmin,
               vmax=vmax)
    plt.title("Residual")
    plt.show()

    print(popt)

    sz_vars, map_vars, bins, Pdl2y, geom = gdi.get_underlying_vars(cluster)
    my_d_ang = map_vars['d_ang']
    gwidth = np.sqrt(abs(popt[3]*popt[4])) # Geometric mean of sigmas.
    sig2fwhm  = np.sqrt(8.0*np.log(2.0))
    mytheta   = gwidth*sig2fwhm
    print(mytheta)
    myradians = mytheta * np.pi/180.0/3600.0
    mydiam    = (my_d_ang * myradians).to('kpc')
    print(Pdl2y)
    Comptony  = popt[0] / sz_vars['tSZ']
    myPres    = popt[0] / myradians / Pdl2y
    print(myPres)

    T_exp     = 5.0 * u.keV
    edens     = (0.2 * u.keV / u.cm**3 / T_exp).decompose()
    tau       = (edens * mydiam * sz_vars['thom_cross']*u.cm**2).decompose()
    beta      = (popt[0]*1.23/sz_vars['tcmb']) / tau
    vpec      = (beta*sz_vars['c']*u.m/u.s).to('km s**-1')
    
    import pdb;pdb.set_trace()

    hdu0 = fits.PrimaryHDU(z_fit,header=hdr)
    hdu0.header.append(("Title","Gaussian Map"))
    hdu0.header.append(("Target",cluster))          
    hdu0.name = 'Gauss_Map'
    myhdu = [hdu0]
    hdulist = fits.HDUList(myhdu)
    outdir = outdir_by_cluster(cluster)
    fullpath = outdir+cluster+'_'+reduc+'_'+filename+'.fits'
    hdulist.writeto(fullpath,overwrite=True)

    
    #mysnrpix = mysnr[gi]
    #myimgpix = myimg[gi]

    #mygi     = (mysnrpix > 3.0)
    #mypixsum = np.sum(myimgpix[mygi])
    
    

def get_xymap(map,pixsize,xcentre=[],ycentre=[],oned=True):
    """
    Returns a map of X and Y offsets (from the center) in arcseconds.

    INPUTS:
    -------
    map      - a 2D array for which you want to construct the xymap
    pixsize  - a quantity (with units of an angle)
    xcentre  - The number of the pixel that marks the X-centre of the map
    ycentre  - The number of the pixel that marks the Y-centre of the map

    """

    ny,nx=map.shape
    ypix = pixsize.to("arcsec").value # Generally pixel sizes are the same...
    xpix = pixsize.to("arcsec").value # ""
    if xcentre == []:
        xcentre = nx/2.0
    if ycentre == []:
        ycentre = ny/2.0

    #############################################################################
    ### Label w/ the transpose that Python imposes?
    #y = np.outer(np.zeros(ny)+1.0,np.arange(0,xpix*(nx), xpix)- xpix* xcentre)   
    #x = np.outer(np.arange(0,ypix*(ny),ypix)- ypix * ycentre, np.zeros(nx) + 1.0)
    #############################################################################
    ### Intuitive labelling:
    x = np.outer(np.zeros(ny)+1.0,np.arange(0,xpix*(nx), xpix)- xpix* xcentre)   
    y = np.outer(np.arange(0,ypix*(ny),ypix)- ypix * ycentre, np.zeros(nx) + 1.0)

    if oned == True:
        x = x.reshape((nx*ny)) #How important is the tuple vs. integer?
        y = y.reshape((nx*ny)) #How important is the tuple vs. integer?

    
    return x,y

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
    pixs=0
    if 'CD1_1' in hdr.keys():
        ras = dxa*hdr['CD1_1'] + dya*hdr['CD2_1'] + hdr['CRVAL1']
        decs= dxa*hdr['CD1_2'] + dya*hdr['CD2_2'] + hdr['CRVAL2']
        pixs= abs(hdr['CD1_1'] * hdr['CD2_2'])**0.5 * 3600.0
    if 'PC1_1' in hdr.keys() or pixs == 0 :
        pcmat = w.wcs.get_pc()
        ras = dxa*pcmat[0,0]*hdr['CDELT1'] + \
              dya*pcmat[1,0]*hdr['CDELT2'] + hdr['CRVAL1']
        decs= dxa*pcmat[0,1]*hdr['CDELT1'] + \
              dya*pcmat[1,1]*hdr['CDELT2'] + hdr['CRVAL2']
        pixs= abs(pcmat[0,0]*hdr['CDELT1'] * \
                  pcmat[1,1]*hdr['CDELT2'])**0.5 * 3600.0
    #if pixs == 0 and 'CDELT1':
                
    pixs = pixs*u.arcsec
    ### Do I want to make ras and decs Angle objects??
    ras  = ras*u.deg; decs = decs*u.deg 
    
    return ras, decs, pixs

def my2dGauss((x,y),amplitude,x0,y0,xstd,ystd,theta,mnlvl):

    xrot = (x-x0) * np.cos(theta) + (y-y0)*np.sin(theta)
    yrot = (y-y0) * np.cos(theta) - (x-x0)*np.sin(theta)

    z = np.exp(-(xrot)**2 / (2*xstd**2) -(yrot)**2 / (2*ystd**2) )*amplitude

    g = z.ravel() + mnlvl

    return g

    
def plot_both_profs(dataset='M2',cluster='Zw3146',model='NP',version='Comparison'):
    
    import PlotFittedProfile as PFP
    PFP=reload(PFP)
    import pickle
    pickdir  = '/home/romero/Results_Python/MUSTANG2/zw3146/'
    pickname = '0703_MUSTANG2_17_Dim_Real_6000S_2000B_ML-NO_PP-NO_POWER_50W_pickle.sav'
    filehandler  = open(pickdir+pickname, 'r')
    myobj1       = pickle.load(filehandler)
    myobj2       = pickle.load(filehandler)
    myobj3       = pickle.load(filehandler)
    filehandler.close()

    solns2 = myobj3.solns
    rads   = myobj2.cfp.bulkarc[0] * 3600.0*180.0/np.pi
    

    outdir = '/home/romero/Results_Python/Rings/'+cluster+'/'
    lv = 'NP-Mar6_v2-Iter5'
    prename = dataset+'_'+cluster+'_'+lv+'_'
    presname = prename+'_fitted_pressure_profile_comparison.png'

    solns = np.load(outdir+'Iter/'+prename+'Solutions.npy')
    #goodY = np.load(outdir+prename+'IntegratedYs.npy')

    

    PFP.plot_pres_bins(solns,dataset,outdir,presname,cluster=cluster,
                       IntegratedYs=None,overlay='XMM',mymodel=model,solns2=solns2,rads2=rads)

    import pdb;pdb.set_trace()
