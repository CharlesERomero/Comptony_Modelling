import numpy as np
from astropy.io import fits
from astropy import coordinates
import scipy.ndimage
from astropy import wcs
from astropy.coordinates import Angle
from astropy.modeling import models, fitting
import warnings, os
import astropy.units as u             # U just got imported!
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as opt
import get_data_info as gdi
from astropy.visualization.wcsaxes import WCSAxes       
import mk_xray_unsharp as mxu
import fit_to_rings as ftr
import pickle
import PlotFittedProfile as PFP
import image_filtering as imf

cpix=0

def get_centroid(cluster):

    if cluster == 'Zw3146':
        ra    = Angle('10h23m39.3336s') # From latest fits (March 1st, 2019)
        dec   = Angle('4d11m14.1248s')  # From latest fits (March 1st, 2019)
    if cluster == 'MACS0717':
        #ra    = Angle('7h17m33.7950s') # By eye, estimate
        #dec   = Angle('37d45m0.226s')  # By eye, estimate
        #ra    = Angle('109.46171284327583d')  # 2D Gaussian Fit
        #dec   = Angle('37.593086726726796d')  # 2D Gaussian Fit
        ra    = Angle('109.39368459d')  # 1D Minkasi Gaussian Fit
        dec   = Angle('37.74642846d')   # 1D Minkasi Gaussian Fit
    if cluster == 'MACS1149':
        #ra    = Angle('11h49m35.2558s') # By eye, estimate
        #dec   = Angle('22d24m6.000s')   # By eye, estimate
        #ra    = Angle('177.50725921389048d') # 2D Gaussian Fit
        #dec   = Angle('22.338484927334466d') # 2D Gaussian Fit
        ra    = Angle('177.50025944d')  # 1D Minkasi Gaussian Fit
        dec   = Angle('22.35894813d')   # 1D Minkasi Gaussian Fit
        
    return ra,dec

def get_xymap_hdr(cluster,img,hdr,rwcs=False,oned=False):

    ras, decs, pixs = astro_from_hdr(hdr)
    ra0, dec0 = get_centroid(cluster)
    w         = wcs.WCS(hdr)
    mypix     = w.wcs_world2pix(ra0.to('degree'),dec0,cpix)
    print 'Center pixels are: ',mypix
    #import pdb;pdb.set_trace()
    xymap     = get_xymap(img,pixs,mypix[0],mypix[1],oned=oned)

    if rwcs:
        return xymap,w,pixs
    else:
        return xymap

def outdir_by_cluster(cluster):

    if cluster == 'Zw3146':
        outdir='/home/data/MUSTANG2/AGBT17_Products/Zw3146/'

    return outdir
        
def find_files(cluster,reduc='IDL',myiter=5,ptmod=False):

    if cluster == 'Zw3146':
        if reduc == 'IDL':
            #mydir = '/home/romero/Results_Python/MUSTANG2/zw3146/ToKeep/'
            #myfile= 'MUSTANG2_Real_Long_Run_PCA_7Mar2019_NP_3asp_v0_ml_pt_fc_MR_C_XferMMv3_MUSTANG2_Models_and_Residuals.fits'
            mydir = '/home/romero/Results_Python/MUSTANG2/zw3146/'
            myfile= 'MUSTANG2_Real_Full_Run_PCA_2019-03-28_NP_2asp_v2_ml_pt_fitGeo_XferMMv3_MUSTANG2_Models_and_Residuals.fits'
            #myfile= 'MUSTANG2_Real_Full_Run_PCA_2019-03-27_NP_2asp_v0_ml_pt_fixEll_XferMMv3_MUSTANG2_Models_and_Residuals.fits'
            if ptmod:
                mext  = 6; snrext=5
            else:
                mext  = 1; snrext=2
        if reduc == 'Minkasi':
            mydir = '/home/romero/Results_Python/Rings/Zw3146/Iter/'
            #myfile= 'M2_Zw3146_NPNP-Mar6_v2-Iter'+repr(myiter)+'Iter1_Residual.fits'
            #myfile= 'M2_Zw3146_NPNP-SVD10_Mar14_ySph_v0-Iter'+repr(myiter)+'_Residual.fits'
            #myfile= 'M2_Zw3146_NPNP-SVD10_Mar15_ySph_normal-Iter'+repr(myiter)+'_Residual.fits'
            #myfile= 'M2_Zw3146_NPNP-SVD10_Mar19_ySph_5c-Iter'+repr(myiter)+'_long_Residual.fits'
            #myfile= 'M2_Zw3146_NPNP-SVD10_Mar19_ySph_10c-Iter'+repr(myiter)+'_long_Residual.fits'
            #myfile= 'M2_Zw3146_NPNP-SVD10_Mar26_ySph_CSB-Iter'+repr(myiter)+'_Residual.fits'
            #myfile= 'M2_Zw3146_NPNP-Mar29_rfwhm_6srcs_ell-Iter'+repr(myiter)+'_Residual.fits' # 2D centroid
            #myfile= 'M2_Zw3146_NPNP-Mar27_rfwhm_6srcs_ell-Iter'+repr(myiter)+'_Residual.fits' # 1D centroid...and other stuff?
            myfile= 'M2_Zw3146_NPNP-SVD10_Mar27_ySph_CSB_6srcs_ell-Iter'+repr(myiter)+'_long_Residual.fits' # 1D centroid
            mext  = 4; snrext=5
        if reduc == 'Minus':
            mydir = '/home/romero/Results_Python/Rings/Zw3146/Iter/'
            myfile= 'M2_Zw3146_NP-Apr_EllRings_5c-Iter'+repr(myiter)+'_DataMinusRings.fits'
            mext  = 4; snrext=5

    return mydir+myfile, mext,snrext
            
def find_orig_files(cluster,reduc='IDL',myiter=5):

    if cluster == 'Zw3146':
        if reduc == 'IDL':
            mydir = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/IDL_Maps/'
            myfile= 'Kelvin_Zw3146_2asp_pca5_qm2_0f08_41Hz_qc_1p2rr_M_PdoCals_dt20_map_iter1.fits'
            mext  = 0; wext=1
        if reduc == 'Minkasi' or reduc == 'Minus':
            mydir = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/Minkasi/Maps/'
            myfile= 'Zw3146_Minkasi_SVD10_Rings_Struct_Map_Pass'+repr(myiter)+'_Mar10.fits'
            mext  =  0; wext=1
    if cluster == 'MACS1149':
        mydir='/home/data/MUSTANG2/AGBT17_Products/MACS1149/'
        if reduc == 'IDL':
            myfile='Kelvin_MACS1149_2asp_pca5_nofiltw_qm2_0f07_41Hz_qc_1p2rr_L_PdoCals_dt20_map_iter1.fits'
            mext = 0; wext=1
    if cluster == 'MACS0717':
        mydir='/home/data/MUSTANG2/AGBT17_Products/MACS0717/'
        if reduc == 'IDL':
            myfile='Kelvin_M0717_2asp_pca5_qm2_0f07_41Hz_qc_1p2rr_L_FebCals_dt20_map_iter1.fits'
            mext = 0; wext=1
        
    return mydir+myfile, mext,wext

def get_maps_file(file,mext,snrext):

    hdu    = fits.open(file)
    hdr    = hdu[mext].header
    img    = hdu[mext].data
    snrhdr = hdu[snrext].header
    snrimg = hdu[snrext].data

    return img,hdr,snrimg,snrhdr

def get_maps_clus(cluster,reduc='IDL',myiter=5,orig=False,ptmod=False):

    if orig:
        file,mext,wext = find_orig_files(cluster,reduc=reduc,myiter=myiter)
        img,hdr,snrimg,snrhdr = get_maps_file(file,mext,wext)    # "SNRIMG, SNRHDR" are for the wtmap
    else:
        file,mext,snrext = find_files(cluster,reduc=reduc,myiter=myiter,ptmod=ptmod)
        img,hdr,snrimg,snrhdr = get_maps_file(file,mext,snrext)
        
    return img,hdr,snrimg,snrhdr

def play_with_resids(cluster='Zw3146',reduc='Minkasi',myiter=5,submed=False,subcnt=False,
                     zoom=6.0,version='_v0',dofit=False,boxsize=1.8):

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

    print('Medians are: ',imgmed,snrmed)
    
    myimg = img.copy()
    mysnr = snrimg.copy()

    if reduc == 'Minkasi' or 'Minus':
        myimg -= imgmed
        mysnr -= snrmed

        estra  =  Angle('10h23m38.8266s') # From latest fits (March 1st, 2019)
        estdec =  Angle('4d11m03.755s')  # From latest fits (March 1st, 2019)
        
        estra1  =  Angle('10h23m40.6600s') # From latest fits (March 1st, 2019)
        estdec1 =  Angle('4d11m27.205s')  # From latest fits (March 1st, 2019)
    else:
        #estra  =  Angle('10h23m37.6590s') # From latest fits (March 1st, 2019)
        #estdec =  Angle('4d11m12.557s')  # From latest fits (March 1st, 2019)
        estra  =  Angle('10h23m38.7437s') # From latest fits (March 1st, 2019)
        estdec =  Angle('4d11m02.998s')  # From latest fits (March 1st, 2019)

        #estra1  =  Angle('10h23m41.4188s') # From latest fits (March 1st, 2019)
        #estdec1 =  Angle('4d11m29.396s')  # From latest fits (March 1st, 2019)
        estra1  =  Angle('10h23m41.3511s') # From latest fits (March 1st, 2019)
        estdec1 =  Angle('4d11m30.264s')  # From latest fits (March 1st, 2019)
       
    ra0,dec0 = get_centroid(cluster)
    cluscens  = [(ra0,dec0)]

    x0,y0 = w.wcs_world2pix(estra.to('degree'),estdec,cpix) # Relative to map...as Python thinks
    x0 += np.min(x)/pixs.value;    y0 += np.min(y)/pixs.value
    x0 *= pixs.value; y0*=pixs.value

    x1,y1 = w.wcs_world2pix(estra1.to('degree'),estdec1,cpix) # Relative to map...as Python thinks
    x1 += np.min(x)/pixs.value;    y1 += np.min(y)/pixs.value
    x1 *= pixs.value; y1*=pixs.value

    xcen,ycen = w.wcs_world2pix(ra0.to('degree'),dec0,cpix)
    xcen += np.min(x)/pixs.value;    ycen += np.min(y)/pixs.value
    
    xstd = 3.0;         ystd = 3.0          # Arcseconds
    theta = 1.0                             # Radians. How should I know?

    myx   = x - xcen;  myy = y - ycen
    myr   =  np.sqrt(myx*myx +myy*myy)
    besti = (myr < 45.0)
    jfc   = (myr > 50.0)

    fimg  = img.copy(); snrcp = snrimg.copy()
    cntmed      = np.median(fimg[besti])
    snrcnt      = np.median(snrcp[besti])
    fimg[besti] = 0.0
    fimg[jfc]   = 0.0
    myannulus   = (fimg > 0)
    anmed       = np.median(fimg[myannulus])

    if subcnt:
        img    -= cntmed
        snrimg -= snrcnt
    myamp = np.max(img[besti])

    if myamp < 0:
        print('hi')
        #import pdb;pdb.set_trace()

    mnlvl = 0.000001

    vmin = np.min(img[gi])
    vmax = np.max(img[gi])
    vmax = (vmax-vmin)/2.0
    vmin = -vmax 
    z = img.copy()
    zin = z[besti]
    fit_p = fitting.LevMarLSQFitter()
    rfi='/home/data/CARMA/ZW3146_USB_10asec.fits'; rrms=7.383e-05

    gpdir = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/Minkasi/GaussPars/'
    gps   = np.load(gpdir+'zwicky_6src_gaussp_3Feb2019_minchi_all_v2.npy')
    cpt   = gps[4:6]
    #import pdb;pdb.set_trace()
    cptloc = [cpt*180.0/np.pi * u.degree]
    
    xpt,ypt = w.wcs_world2pix(cptloc[0][0],cptloc[0][1],cpix)
    xpt += np.min(x)/pixs.value;    ypt += np.min(y)/pixs.value
    xpt *= pixs.value; ypt *= pixs.value
    myfilename = 'MUSTANG2_Residual_Map_'+reduc+version+'_Iter'+repr(myiter)
    
    plot_image(img,hdr,w,pixs.value,cluster,cimg=snrimg,logscale=False,format='png',myfontsize=15,
               vmin=vmin,vmax=vmax,ptcens=cptloc,
               micro=True,spectral=False,cbar=True,filename=myfilename,
               boxsize=boxsize,zoom=1.0,submed=submed,rfi=rfi,rrms=rrms,cluscens=cluscens)
    # Boxsize=1.8 was the default for other images made.

    if dofit:
        initial_guess = (myamp,x0,y0,xstd,ystd,theta,mnlvl)
        popt, pcov = opt.curve_fit(my2dGauss, (x[besti], y[besti]), zin, p0=initial_guess)
    
        initial_guess2 = (myamp,x0,y0,myamp,x1,y1,xstd,ystd,theta,mnlvl)
        sigx = 10.0
        #mnbnd = np.inf if reduc == 'Minkasi' else 1.0e-6
        mnbnd = 1.0e-6
        bounds = ([0.0,x0-sigx,y0-sigx,0.0,x1-sigx,y1-sigx,2.0,2.0,0.0,-mnbnd],
                  [myamp*2,x0+sigx,y0+sigx,myamp*2,x1+sigx,y1+sigx,16.0,16.0,2*np.pi,mnbnd])
        #print(bounds)
        popt2, pcov2 = opt.curve_fit(mylinked_2dGauss, (x[besti], y[besti]), zin, p0=initial_guess2,bounds=bounds)

        print(popt2)

        dr0 = np.sqrt( (popt2[1] - xpt)**2 + (popt2[2] - ypt)**2 )
        dr1 = np.sqrt( (popt2[4] - xpt)**2 + (popt2[5] - ypt)**2 )
        print('Radial seperation: ',dr0, dr1)

        myx = (popt2[1] - np.min(x))/pixs.value;   myy = (popt2[2] - np.min(y))/pixs.value
        ra_0, dec_0 = w.wcs_pix2world(myx,myy,cpix)
        #print('SW WCS Coordinates: ',ra_0.to_string(unit=u.hour),dec_0.to_string(unit=u.degree))
        rastr = make_sexagesimal(ra_0,ishours=True); decstr = make_sexagesimal(dec_0)
        print('SW WCS Coordinates: ',rastr,decstr)

        myx = (popt2[4] - np.min(x))/pixs.value;   myy = (popt2[5] - np.min(y))/pixs.value
        ra_0, dec_0 = w.wcs_pix2world(myx,myy,cpix)
        #print('NE WCS Coordinates: ',ra_0.to_string(unit=u.hour),dec_0.to_string(unit=u.degree))
        rastr = make_sexagesimal(ra_0,ishours=True); decstr = make_sexagesimal(dec_0)
        print('SW WCS Coordinates: ',rastr,decstr)
        
        s2f  = np.sqrt(8.0 * np.log(2.0))
        fwhm1 = s2f*popt2[6];    fwhm2 = s2f*popt2[7]
        print(fwhm1,fwhm2, popt2[8]*180.0/np.pi)
    
        #import pdb;pdb.set_trace()
    
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
            #filename=myfilename+'_zoom'+version+'_Iter'+repr(myiter)
            myfilename = myfilename+'_zoom'
            xl,xu = int(xsz*myf),int(xsz*(1-myf))
            yl,yu = int(ysz*myf),int(ysz*(1-myf))

        #p_fit  = my2dGauss((x, y),popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]) 
        #p_noml = my2dGauss((x, y),popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],0) 
        p_fit  = mylinked_2dGauss((x, y),popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7],popt2[8],popt2[9]) 
        p_noml = mylinked_2dGauss((x, y),popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7],popt2[8],0) 
        z_fit  = p_fit.reshape(x.shape)   # Gaussian with    the mean level
        z_noml  = p_noml.reshape(x.shape) # Gaussian without the mean level

        print('-----------------------------------------------------------------')
        print('Amplitude (uK):           ',popt[0]*1.0e6)
        print('Delta RA  (arcseconds):   ',popt[1])
        print('Delta Dec (arcseconds):   ',popt[2])
        print('Axis A sigma ("):         ',popt[3])
        print('Axis B sigma ("):         ',popt[4])
        print('Rotation angle (radians): ',popt[5])
        print('Mean level (uK):          ',popt[6]*1.0e6)
        print('-----------------------------------------------------------------')
        myfilename = 'MUSTANG2_Residual_Map_'+reduc+version+'_Iter'+repr(myiter)+'_wModel'
        
        plot_image(img,hdr,w,pixs.value,cluster,cimg=snrimg,logscale=False,format='png',myfontsize=15,
                   vmin=vmin,vmax=vmax,modimg=z_noml,ptcens=cptloc,
                   micro=True,spectral=False,cbar=True,filename=myfilename,
                   boxsize=boxsize,zoom=1.0,submed=submed,rfi=rfi,rrms=rrms,cluscens=cluscens)
        # Boxsize=1.8 was the default for other images made.

        plt.close()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(z[xl:xu,yl:yu], origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(z_fit[xl:xu,yl:yu], origin='lower', interpolation='nearest', vmin=vmin,
                   vmax=vmax)
        plt.title("Model")
        plt.subplot(1, 3, 3)
        plt.imshow(z[xl:xu,yl:yu] - z_noml[xl:xu,yl:yu], origin='lower', interpolation='nearest', vmin=vmin,
                   vmax=vmax)
        plt.title("Residual")
        plt.show()

        sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)
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
        
        #import pdb;pdb.set_trace()

        hdu0 = fits.PrimaryHDU(z_fit,header=hdr)
        hdu0.header.append(("Title","Gaussian Map"))
        hdu0.header.append(("Target",cluster))          
        hdu0.name = 'Gauss_Map'
        myhdu = [hdu0]
        hdulist = fits.HDUList(myhdu)
        outdir = outdir_by_cluster(cluster)
        fullpath = outdir+cluster+'_'+myfilename+'.fits'
        #myfilename = 'MUSTANG2_Residual_Map_'+reduc+version+'_Iter'+repr(myiter)

        hdulist.writeto(fullpath,overwrite=True)
        
    else:
        plt.show()

 

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
    x = np.outer(np.zeros(ny)+1.0,np.arange(nx)*xpix- xpix* xcentre)   
    y = np.outer(np.arange(ny)*ypix- ypix * ycentre, np.zeros(nx) + 1.0)

    #import pdb;pdb.set_trace()
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

def mylinked_2dGauss((x,y),amp0,x0,y0,amp1,x1,y1,xstd,ystd,theta,mnlvl):

    xrot0 = (x-x0) * np.cos(theta) + (y-y0)*np.sin(theta)
    yrot0 = (y-y0) * np.cos(theta) - (x-x0)*np.sin(theta)
    z0 = np.exp(-(xrot0)**2 / (2*xstd**2) -(yrot0)**2 / (2*ystd**2) )*amp0

    xrot1 = (x-x1) * np.cos(theta) + (y-y1)*np.sin(theta)
    yrot1 = (y-y1) * np.cos(theta) - (x-x1)*np.sin(theta)
    z1 = np.exp(-(xrot1)**2 / (2*xstd**2) -(yrot1)**2 / (2*ystd**2) )*amp1

    z = z1 + z0
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

def plot_image(img,hdr,mywcs,pixs,src,cimg=None,logscale=False,format='png',myfontsize=15,
               vmin=None,vmax=None,modimg=None,
               micro=False,spectral=False,cbar=True,filename='MUSTANG2_Residual_Map',
               boxsize=None,zoom=1.0,submed=True,cbtit='K',title=' ',rfi=None,rrms=1.0,
               cluscens=[],ptcens=[],xrcen=[]):

    nlabel=0
    mew   = 2   # Marker edge width, for centroids
    ms    = 10  # Marker size, for centroids
    
    if format == 'png':
        #fig = plt.figure(figsize=(16,16),dpi=300) 
        fig = plt.figure(figsize=(8,7),dpi=300) 
        myfontsize/=2.0
    else:
        fig = plt.figure(figsize=(3.75,3.5),dpi=200)
        myfontsize/=4.5
        mew/=2
        ms/=2
        

    axpos=[0.23, 0.14, 0.7, 0.8]
    if cbar ==  False:
        axpos=[0.12, 0.1, 0.8, 0.8]
    ax = WCSAxes(fig, axpos, wcs=mywcs)
    fig.add_axes(ax)  # note that the axes have to be added to the figure

    if len(cluscens) > 0:
        cx,cy = x_centroids(cluscens,mywcs,mycolor='r',mylabel='SZ centroid',
                            mew=mew,ms=ms,ax=ax)
        nlabel+=1

        
    
    pimg = img*1.0
    if (vmin is None):
        vmin = -0.0005; vmax = 0.0005
    
    if micro:
        pimg  = img*1.0e6
        vmax *= 1.0e6
        vmin *= 1.0e6  
        cbtit=r'$\mu$'+cbtit

        
    if not (boxsize is None):
        boxpix  = int(boxsize*60.0/pixs)
        xsz,ysz = img.shape
        xl,xu = xsz/2 - boxpix/2, xsz/2 + boxpix/2
        yl,yu = ysz/2 - boxpix/2, ysz/2 + boxpix/2
        immed = np.median(pimg[xl:xu,yl:yu])
        ax.set_xlim(xl,xu)
        ax.set_ylim(yl,yu)
        filename=filename+'_box'
        #print(xl,xu,yl,yu)
        zoom=-10

    if zoom > 1:
        xsz,ysz = img.shape
        myf     = 0.5 - 1.0/(2.0*zoom)
        #print myf,1-myf
        ax.set_xlim(xsz*myf,xsz*(1-myf))
        ax.set_ylim(ysz*myf,ysz*(1-myf))
        #ax.set_xlim(xsz/4,3*xsz/4)
        #ax.set_ylim(ysz/4,3*ysz/4)
        filename=filename+'_zoom'
        xl,xu = int(xsz*myf),int(xsz*(1-myf))
        yl,yu = int(ysz*myf),int(ysz*(1-myf))
        immed = np.median(pimg[xl:xu,yl:yu])

    if submed: pimg -= immed
    
    if spectral:
        mycmap = cm.get_cmap('nipy_spectral')
    else:
        mycmap = cm.get_cmap('RdBu_r')
        
    if logscale == False:
        cax = ax.imshow(pimg,cmap=mycmap,origin='lower',
                        vmin=vmin,vmax=vmax)
    else:
        cax = ax.imshow(pimg,cmap=mycmap,origin='lower',
                    norm=SymLogNorm(linthresh=vmin,vmin=-vmin, vmax=vmax))

    #print(vmin,vmax,np.min(pimg),np.max(pimg))
    #import pdb;pdb.set_trace()

        
    if cbar == True:
        mycb = fig.colorbar(cax)
        mycb.ax.tick_params(labelsize=myfontsize) 
        mycb.set_label(cbtit,fontsize=myfontsize)
#        mycb.ax.set_label("Jy/beam")
    ax.set_title(title,fontsize=myfontsize*2)

    if type(rfi) != type(None):
        szri,szhd = mxu.get_map_and_hdr(rfi,ext=0)
        sz_img    = szri.squeeze()
        szhd['NAXIS'] = 2
        if 'NAXIS3' in szhd.keys(): szhd.pop('NAXIS3')
        if 'NAXIS4' in szhd.keys(): szhd.pop('NAXIS4')
        if 'CRPIX3' in szhd.keys(): szhd.pop('CRPIX3')                                        
        if 'CDELT3' in szhd.keys(): szhd.pop('CDELT3')                                          
        if 'CRVAL3' in szhd.keys(): szhd.pop('CRVAL3')                                        
        if 'CTYPE3' in szhd.keys(): szhd.pop('CTYPE3')                                                 
        if 'CRPIX4' in szhd.keys(): szhd.pop('CRPIX4')                                        
        if 'CDELT4' in szhd.keys(): szhd.pop('CDELT4')                                          
        if 'CRVAL4' in szhd.keys(): szhd.pop('CRVAL4')                                        
        if 'CTYPE4' in szhd.keys(): szhd.pop('CTYPE4')                                                 
        if 'HISTORY' in szhd.keys(): szhd.pop('HISTORY')                                                 

        #import pdb;pdb.set_trace()
        sz_image  = mxu.grid_map2_onto_map1(hdr,img,szhd,sz_img)

        myrad = 25
        xysz  = sz_image.shape
        xxx   = np.outer(np.arange(xysz[0]),np.ones(xysz[1])) - cx[0]
        yyy   = np.outer(np.ones(xysz[0]),np.arange(xysz[1])) - cy[0]

        rmap  = (xxx**2 + yyy**2)**0.5

        rexcl = (rmap > myrad)
        sz_image[rexcl] /= 10.0
        
        sz_image *= -1.0
        sz_levels = rrms*np.arange(3,8)
        szset     = plt.contour(sz_image, sz_levels, colors='g')

    
    if type(cimg) != type(None):
        snrmed  = np.median(cimg)
        if zoom > 1 or zoom == -10: snrmed = np.median(cimg[xl:xu,yl:yu])
        if submed: cimg -= snrmed
        #max_snr = int(np.max(cimg));        min_snr = int(np.min(cimg))
        max_snr  = int(np.max(cimg)/2)*2;        min_snr = int(np.min(cimg)/2)*2
        snr_span = (max_snr-min_snr)/2 + 2
        snrlevels= range(min_snr,0,2)
        snrlevels.extend(range(2,snr_span*2+min_snr,2))
        #import pdb;pdb.set_trace()
        #snr_set  = set(range(min_snr,snr_span*2+min_snr,2)) - set([0])
        #snrlevels= list(np.sort(np.asarray(snr_set)))
        print(snrlevels)
        cset    = plt.contour(cimg, snrlevels, colors='k')

    if type(modimg) != type(None):
        min_mod = 1.28e-5; max_mod = 10.0*min_mod
        modlevels= list(np.arange(min_mod,max_mod,min_mod))
        cset    = plt.contour(modimg, modlevels, colors='c')

        
    if len(ptcens) > 0:
        x_centroids(ptcens,mywcs,mycolor='y',  mylabel='MUSTANG2 pts.',
                    mew=mew*1.5,ms=ms,ax=ax)
        nlabel+=1
        
    if len(xrcen) > 0:
        x_centroids(xrcen,mywcs,mycolor='m',  mylabel='X-ray centroid',
                    mew=mew,ms=ms,ax=ax)
        nlabel+=1
        
    add_herschel_locs(mywcs,ax,wl=500, mew=mew, ms=ms)

        
    if nlabel > 0:
        if src == 'Zw3146':
            print 'No legend for you'
            #plt.legend(loc='upper right',fontsize=myfontsize)
        else:
            plt.legend(loc='lower right',fontsize=myfontsize)

        
    cwd = os.getcwd(); fullbase = os.path.join(cwd,filename)
    fulleps = fullbase+'.eps'; fullpng = fullbase+'.png'
    #plt.tight_layout(fig)
    
    if format == 'png':
        plt.savefig(fullpng,format='png')
    else:
        plt.savefig(fulleps,format='eps')
   
def x_centroids(centroids,w,mycolor='k',ms=10,mew=2,mylabel=None,ax=None):

    xs=[];ys=[]
    for centroid in centroids:
        xc,yc   = w.wcs_world2pix(centroid[0].to('deg'),centroid[1].to('deg'),cpix)
        #print 'Your coordinates are: ',xc,' and ',yc
        xs.extend([np.asscalar(xc)])
        ys.extend([np.asscalar(yc)])

    if ax is None:
        plt.plot(xs,ys,"x",color=mycolor,markersize=ms,mew=mew,label=mylabel)
    else:
        ax.plot(xs,ys,"x",color=mycolor,markersize=ms,mew=mew,label=mylabel)
        
        
    print 'X marks the spot(s).'
    return xs,ys

def make_sexagesimal(tdeg,ishours=False):

    hours = tdeg/15.0if ishours else tdeg

    hrs   = np.floor(hours)
    mins  = np.floor( (hours - hrs)*60.0)
    secs  = ( (hours - hrs)*60.0 - mins)*60.0

    sh    = "{0:02d}".format(int(hrs))
    sm    = "{0:02d}".format(int(mins))
    ss    = "{0:.2f}".format(secs)

    print(sh,':',sm,':',ss)

    #import pdb;pdb.set_trace()

    mystring = sh+':'+sm+':'+ss
    
    return mystring

def make_ptsrc_map(ptamps,xymap,header,gpfile=None,pixs=2.0):

    if gpfile is None:
        gpdir = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/Minkasi/GaussPars/'
        gpfile = gpdir+'zwicky_6src_gaussp_3Feb2019_minchi_all_v2.npy'
        
    gps    = np.load(gpfile)
        
    ra     = gps[4::4] * 180.0 / np.pi
    raang  = Angle(ra, unit=u.deg)
    #print(raang.to_string(unit=u.hour))
    dec    = gps[5::4] * 180.0 / np.pi
    decang = Angle(dec, unit=u.deg)
    #print(decang.to_string(unit=u.degree))

    sigma    = gps[6::4] * 180.0 / np.pi * 3600.0
    sig2fwhm = np.sqrt(8.0*np.log(2))
    fwhm     = sigma*sig2fwhm
    #print(fwhm)
    
    #amp    = gps[3::4]
    #mJy    = amp*1.0e3 * Jy2K * (fwhm/10.735)**2
    #print(mJy)

    w      = wcs.WCS(header)
    x,y    = xymap
    myx    = x.copy(); myx /= 2; myx -= (min(myx)-1)
    myy    = y.copy(); myy /= 2; myy -= (min(myy)-1)
    ptmap  = x.copy(); ptmap *= 0.0
    for ra0,dec0,sig,amp in zip(raang,decang,sigma,ptamps):
        mypix  = w.wcs_world2pix(ra0,dec0,1)
        xt, yt = myx-mypix[0], myy-mypix[1]
        ptmap += amp * np.exp(- (xt**2 + yt**2)/(2.0*(sig/pixs)**2))

    return ptmap

def fit_2dgauss(cluster='Zw3146',reduc='Minkasi',myiter=5,filename='GaussResid',
                zoom=6.0):

    if reduc == 'Minkasi' or reduc == 'IDL':
        img,hdr,wtmap,wthdr = get_maps_clus(cluster,reduc=reduc,myiter=myiter,orig=True)
    if reduc == 'XMM':
        xrf='/home/data/X-ray/XMM/mosaic_ZW3146_asmooth.fits'
        img, hdr = fits.getdata(xrf, 0, header=True)
        #import pdb;pdb.set_trace()
    if reduc == 'CXO':
        xrf='/home/data/X-ray/CXO/Zw3146/srcfree_bin4_500-4000_band1_flux.fits'
        img, hdr = fits.getdata(xrf, 0, header=True)
        #import pdb;pdb.set_trace()
        
    xymap,w,pixs = get_xymap_hdr(cluster,img,hdr,rwcs=True,oned=True)
    x,y = xymap
    r = np.sqrt(x*x +y*y)

    curve,deriv,edges,trim,nsrc = ftr.load_rings(option='M2',cluster=cluster,session=0,iteration=myiter)
    if reduc == 'Minkasi':
        #curve,deriv,edges,trim,nsrc = ftr.load_rings(option='M2',cluster=cluster,session=0,iteration=myiter)
        ptamps = ftr.get_data_cov_edges(curve,deriv,edges,trim=trim,slices=False,nslice=0,slicenum=0,
                                nsrc=6,fpause=False,ptsrcamps=True)
        myamp = -8.0e-4
    elif reduc == 'IDL':
        m2sd = '/home/romero/Results_Python/MUSTANG2/zw3146/ToKeep/'
        m2sf = m2sd+'0703_MUSTANG2_17_Dim_Real_6000S_2000B_ML-NO_PP-NO_POWER_50W_pickle.sav'
        handle = open(m2sf,'rb')
        dv = pickle.load(handle)
        hk = pickle.load(handle)
        efv = pickle.load(handle)
        pos = efv.solns[:,0]       # Correct units...the unitless kind, mostly
        ptamps = pos[-6:]
        myamp = -5.0e4
    else:
        myamp = 0.0016 if reduc == 'XMM' else 0.00003

    if reduc =='XMM' or reduc=='CXO':
        ptsub = img.copy()
        p2d   = ptsub*1.0
        pflat = ptsub.flatten()
        if reduc == 'CXO':
            gi = (pflat != 0.0)
            pflat = pflat[gi]; x = x[gi]; y = y[gi];  r = r[gi]
    else:
        ptmap = make_ptsrc_map(ptamps,xymap,hdr)
        ptsub = img - ptmap.reshape(img.shape)
        smfw = 10.0 * u.arcsec
        sigma = (smfw/pixs).value/(2.0*np.sqrt(2.0*np.log(2.0)))
        pflat = scipy.ndimage.filters.gaussian_filter(ptsub,sigma)
        p2d   = pflat*1.0
        pflat = pflat.flatten()

    x0   = 0.0 ;    y0   = 0.0
    xstd = 25.0;    ystd = 20.0
    theta = 0.5
    mnlvl = 4.0e-6

    ellipt = np.zeros(len(edges))
    thetas = np.zeros(len(edges))
    avgr   = (edges[:-1] + edges[1:])/2.0

    #initial_guess = (myamp,x0,y0,xstd,ystd,theta,mnlvl)
    initial_guess = [myamp,x0,y0,xstd,ystd,theta,mnlvl]
    besti = (r < 180.0)
    popt, pcov = opt.curve_fit(my2dGauss, (x[besti], y[besti]), pflat[besti], p0=initial_guess)
    print(popt)
    myx = (popt[1] - np.min(x))/pixs; myy = (popt[2] - np.min(y))/pixs
    myra,mydec = w.wcs_pix2world(myx,myy,0)
    print(myra,mydec)
    
    
    #import pdb;pdb.set_trace()

    
    for i,(inner,outer) in enumerate(zip(edges[0:-1],edges[1:])):
        goodi = (r >= inner)
        alsoi = (r < outer)
        besti = [a and b for a,b in zip(goodi,alsoi)]
        #initial_guess = (myamp,x0,y0,xstd,ystd,theta,mnlvl)
        print(i,inner,outer,initial_guess)
        #avgr = (inner+outer)/2.0
        try:
            popt, pcov = opt.curve_fit(my2dGauss, (x[besti], y[besti]), pflat[besti], p0=initial_guess)
            #print('I hate your face')
            print(popt)
            ellipt[i] = popt[4]/popt[3]
            thetas[i] = popt[5] % np.pi
        except:
            mnval  = np.mean(pflat[besti])
            stdev  = np.std(pflat[besti])
            width  = outer - inner
            ni1    = (r >= inner - width)
            ni2    = (r < outer + width)
            gi     = [a and b for a,b in zip(ni1,ni2)]

            stdcut = 1.0
            mypi = gi
            myx    = x[gi]
            myy    = y[gi]
            #while np.sum(mypi) > np.sqrt(np.sum(besti)):
            mymin = np.max([np.sum(besti)/5,5])
            while np.sum(mypi) > mymin*2:
                mypi1  = (pflat[gi] > mnval - stdev*stdcut) 
                mypi2  = (pflat[gi] < mnval + stdev*stdcut) 
                mypi   = [a and b for a,b in zip(mypi1,mypi2)]
                myx    = x[gi][mypi]
                myy    = y[gi][mypi]
                print np.sum(mypi)
                stdcut /= 2.0
            ell_guess = [1.0,1.0,inner+1.0,outer,1.0]
            print(ell_guess,myx.shape)
            try:
                popt, pcov = opt.curve_fit(my_ellipse, (myx,myy), np.ones(myx.shape), p0=ell_guess)
            except:
                import pdb;pdb.set_trace()
                popt = ell_guess
            myr = (myx**2 + myy**2)**0.5
            myt = np.arctan2(myy,myx)

            fitr = np.zeros(myr.shape) + np.mean(myr)
            fitt = np.arange(len(myr))*2.0*np.pi/len(myr) - np.pi
            fitx = np.cos(fitt - popt[4]) * popt[2]
            fity = np.sin(fitt - popt[4]) * popt[3]
            rfit = (fitx**2 + fity**2)**0.5
            
            ellipt[i] = popt[3]/popt[2]
            thetas[i] = popt[4] % np.pi

            #plt.plot(myt,myr,'o')
            #plt.plot(fitt,rfit)
            #plt.show()
            #plt.imshow(p2d,origin='lower')
            #plt.show()
            
    import pdb;pdb.set_trace()
    mygi1 = np.isfinite(ellipt)
    mygi2 = (ellipt > 0)
    mygi  = [a and b for a,b in zip(mygi1,mygi2)]
    plt.plot(avgr[mygi],ellipt[mygi])
    plt.ylim((0,1.5))
    plt.show()

    ECdir = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/EllipticityChecks/'
    np.save(ECdir+reduc+'_Ellipticity_Rads.sav',avgr[mygi])
    np.save(ECdir+reduc+'_Ellipticity_Ells.sav',ellipt[mygi])
    #import pdb;pdb.set_trace()

    print('hi')

def my_ellipse((x,y),x0,y0,xstd,ystd,theta):

    xp = (x - x0)
    yp = (y - y0)
    xr = xp*np.cos(theta) + yp*np.sin(theta)
    yr = yp*np.cos(theta) - xp*np.sin(theta)
    z  = xr**2 / xstd**2 + yr**2 / ystd**2

    return z

def my_ell((x,y),xstd,ystd,theta):

    xp = x*1.0
    yp = y*1.0
    xr = xp*np.cos(theta) + yp*np.sin(theta)
    yr = yp*np.cos(theta) - xp*np.sin(theta)
    z  = xr**2 / xstd**2 + yr**2 / ystd**2

    return z

def plot_all_ells():

    ECdir    = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/EllipticityChecks/'
    xmm_rads = np.load(ECdir+'XMM_Ellipticity_Rads.sav.npy')
    xmm_ells = np.load(ECdir+'XMM_Ellipticity_Ells.sav.npy')
    idl_rads = np.load(ECdir+'IDL_Ellipticity_Rads.sav.npy')
    idl_ells = np.load(ECdir+'IDL_Ellipticity_Ells.sav.npy')
    min_rads = np.load(ECdir+'Minkasi_Ellipticity_Rads.sav.npy')
    min_ells = np.load(ECdir+'Minkasi_Ellipticity_Ells.sav.npy')

    maxr = 250.0;pltmax = 120.0
    gxmmr = (xmm_rads < maxr)
    bxmme = (xmm_ells < 2.0)
    bxmmi = [a and b for a,b in zip(gxmmr,bxmme)]
    plt.plot(xmm_rads[bxmmi],xmm_ells[bxmmi],label='XMM')
    gidlr = (idl_rads < maxr)
    bidle = (idl_ells > 0.4)
    gidle = (idl_ells < 2.0)
    gidli = [a and b and c for a,b,c in zip(gidlr,bidle,gidle)]
    plt.plot(idl_rads[gidli],idl_ells[gidli],label='IDL')
    gminr = (min_rads < maxr)
    plt.plot(min_rads[gminr],min_ells[gminr],label='Minkasi')
    plt.ylim((0.4,1.2))
    plt.xlim((0,pltmax))
    plt.ylabel('Minor-to-Major Axis Ratio')
    plt.xlabel('Average Radius')
    plt.legend()
    plt.show()

    #import pdb;pdb.set_trace()

def regbindata(arr1,arr2,arrmin=None,arrmax=None,spacing=None):

    # I should add a check to make sure arr1 and arr2 are flattened.
    
    if arrmin is None:
        arrmin = np.min(arr1)
    if arrmax is None:
        arrmax = np.max(arr1)
    if spacing is None:
        nbin    = np.sqrt(len(arr1))
        spacing = (arrmax-arrmin)/nbin
    
    nbin = int(np.round((arrmax-arrmin)/spacing))
        
    inedges = np.arange(nbin)*spacing + arrmin

    mymny = np.zeros((nbin))
    mystd = np.zeros((nbin))
    mymnx = np.zeros((nbin))
    mycou = np.zeros((nbin))

    for i,inedge in enumerate(inedges):
        cond1 = (arr1 >= inedge)
        cond2 = (arr1 < inedge+spacing)
        cond = np.asarray([c1 and c2 for c1,c2 in zip(cond1,cond2)],dtype=bool)
        myxs = arr1[cond]
        myys = arr2[cond]

        mymny[i] = np.mean(myys)
        mymnx[i] = np.mean(myxs)
        mystd[i] = np.std(myys)
        mycou[i] = len(myys)

    binned = {"inner_edge":inedges,"abs_mean":mymnx,"ord_mean":mymny,
              "ord_std":mystd,"counts":mycou,"spacing":spacing}

    return binned

def radial_plots(cluster='Zw3146',myiter=5,filename='GaussResid',reduc='IDL',slope=True,
                zoom=6.0):

    outdir = outdir_by_cluster(cluster)
    img,hdr,wtmap,wthdr = get_maps_clus(cluster,reduc=reduc,myiter=myiter,orig=True) 
    ptsimg,ptshdr,Bulkmap,Bulkhdr = get_maps_clus(cluster,reduc=reduc,myiter=myiter,ptmod=True) 
    xymap,w,pixs = get_xymap_hdr(cluster,img,hdr,rwcs=True,oned=True)
    x,y = xymap
    r = np.sqrt(x*x +y*y)

    curve,deriv,edges,trim,nsrc = ftr.load_rings(option='M2',cluster=cluster,session=0,iteration=myiter)
    gdata,gcurv,gedge = ftr.get_data_cov_edges(curve,deriv,edges,trim=trim,slices=False,nslice=0,slicenum=0,
                                   nsrc=6,fpause=False)

    m2sd = '/home/romero/Results_Python/MUSTANG2/zw3146/ToKeep/'
    m2sf = m2sd+'0703_MUSTANG2_17_Dim_Real_6000S_2000B_ML-NO_PP-NO_POWER_50W_pickle.sav'
    handle = open(m2sf,'rb')
    dv = pickle.load(handle)
    hk = pickle.load(handle)
    efv = pickle.load(handle)
    pos = efv.solns[:,0]       # Correct units...the unitless kind, mostly
    ptamps = pos[-6:]
    myamp = -5.0e4

    #ptmap = make_ptsrc_map(ptamps,xymap,hdr)
    ptsub = img - ptsimg.reshape(img.shape)
    smfw = 10.0 * u.arcsec
    sigma = (smfw/pixs).value/(2.0*np.sqrt(2.0*np.log(2.0)))
    pflat = scipy.ndimage.filters.gaussian_filter(ptsub,sigma)
    p2d   = pflat*1.0
    pflat = pflat.flatten()

    wtflat = wtmap.flatten()
    nzwind = (wtflat > 0)
    rads   = r[nzwind]
    vals   = pflat[nzwind]
    maxrad = 300.0           # arcseconds
    radspace = 5.0           # arcseconds
    
    binned = regbindata(rads,vals,arrmin=0,arrmax=maxrad,spacing=radspace)

    ###########################################################################
    
    sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)
    if slope:
        #gdata,gcurv,gedge      = get_better_slope_rings(option=dataset,cluster=cluster,session=session)
        slopes = ftr.get_slope(option='M2',cluster=cluster,mypass=myiter,slices=False,gedge=gedge,trim=trim,
                           nsrc=6,fpause=False,session=0)
        print 'Your slopes are: ',slopes
        print '=============================================================='

    #nrings    = len(gedge)-1
    #if slopes is None:
    #    slopes = np.zeros((nrings))
    #edgy_rads = gedge * np.pi / (180.0*3600.0)
    #yProf   = np.zeros(map_vars['thetas'].shape)
    #for i in range(nrings):
    #    gRads1 = (map_vars['thetas'] >=  edgy_rads[i]) # Good radii (condition 1)
    #    gRads2 = (map_vars['thetas'] < edgy_rads[i+1]) # Good radii (condition 2)
    #    gRads  = [gRad1 and gRad2 for gRad1,gRad2 in zip(gRads1,gRads2)]
    #    myrads = map_vars['thetas'][gRads]
    #    a      = 1.0 - slopes[i]*(myrads-edgy_rads[i]) * (3600*180/np.pi)
    #    yProf[gRads] = gdata[i]*a

    #import pdb;pdb.set_trace()
        
    profname = 'IDL_and_Minkasi_brightness_profiles.png'
    PFP.plot_surface_profs_v2(binned['ord_mean'],binned['abs_mean'],gdata,gcurv,gedge,outdir,profname,
                              cluster=cluster,mymodel='NP',pinit=None,bare=False,slopes=slopes)

    
def resid_ps(cluster='Zw3146',reduc='Minkasi',myiter=5,filename='PowerSpectra.png'):

    myfile, mext,snrext = find_files(cluster,reduc=reduc,myiter=5,ptmod=False)
    if reduc == 'Minkasi':
        modelext = 0; rmsext = 6
        nfile = get_map_file(cluster=cluster,Noise=True)
        nimg, nhdr = fits.getdata(nfile, 0, header=True)
        mimg, mhdr = fits.getdata(myfile, modelext, header=True)

    img,hdr,wtmap,wthdr = get_maps_clus(cluster,reduc=reduc,myiter=myiter)
    wcscen = get_centroid(cluster)
    outdir = outdir_by_cluster(cluster)

    mult = 1.0e6
    
    ax=None
    ax = MakePS(img*mult,hdr,wcscen,doplot=True,ax=ax,mylabel='Residual')
    #print(img.shape,mimg.shape)
    #print(mhdr)
    #import pdb;pdb.set_trace()
    ax = MakePS(img/mimg*1.0e2,hdr,wcscen,doplot=True,ax=ax,mylabel='Residual/Model * 100')
    #ax = MakePS(mimg,mhdr,wcscen,doplot=True,ax=ax,mylabel='Model')
    ax = MakePS(nimg*mult,nhdr,wcscen,doplot=True,ax=ax,mylabel='Noise')
    fullpath = os.path.join(outdir,filename)
    ax.legend()
    plt.savefig(fullpath)
    plt.close()

def MakePS(img,hdr,wcscen,xbox=300.0,ybox=300.0,pixsize=2.0,doplot=False,ax=None,mylabel='Unknown'):

    #imft = np.fft.fft2(img)
    
    w         = wcs.WCS(hdr)
    mypix     = w.wcs_world2pix(wcscen[0].to('degree'),wcscen[1].to('degree'),cpix)

    xl        = int(mypix[0]-xbox/2/pixsize);  xu = int(mypix[0]+xbox/2/pixsize)
    yl        = int(mypix[1]-ybox/2/pixsize);  yu = int(mypix[1]+ybox/2/pixsize)
    print(xl,xu,yl,yu)
    #import pdb;pdb.set_trace()
    myimg     = img[xl:xu,yl:yu]
    print(myimg.shape)
    
    kbin,pkmn,pkstd = imf.power_spectrum_2d(myimg,nbins=30)

    if doplot:

        if ax is None:
            myfig = plt.figure(1,figsize=(8,6),dpi=300)
            plt.clf()
            ax = myfig.add_subplot(111)
            ax.set_xlabel(r'Wavenumber (/$^{\prime\prime}$)')
            ax.set_ylabel(r'Spectral Power ($\mu$K$^{2}$)')
            ax.set_yscale('log')
            ax.set_title(r'Power Spectra within a $300^{\prime\prime} \times 300^{\prime\prime}$ box')
            
        ax.errorbar(kbin[1:]/pixsize,pkmn[1:],yerr=pkstd[1:],label=mylabel,capsize=5)

    return ax

def get_map_file(cluster='Zw3146',SNR=False,Noise=False,iteration=5):

    m2dir='/home/data/MUSTANG2/'
    if cluster == 'Zw3146':
      mapdir=m2dir+'AGBT17_Products/Zw3146/Minkasi/Maps/'
      if SNR:
          #mfile = mapdir+'Zw3146_Minkasi_Struct_SNR_Pass5_Feb17.fits'
          #mfile = mapdir+'Zw3146_Minkasi_SVD10_Struct_SNR_Pass10_Mar10.fits'
          mfile = mapdir+'Zw3146_Minkasi_SVD10_Rings_Struct_SNR_Pass'+str(iteration)+'_Mar10.fits'
      elif Noise:
          #mfile = mapdir+'Zw3146_Minkasi_Struct_Noise_Pass5_Feb17.fits'
          #mfile = mapdir+'Zw3146_Minkasi_SVD10_Struct_Noise_Pass10_Mar10.fits'
          mfile = mapdir+'Zw3146_Minkasi_SVD10_Rings_Struct_Noise_Pass'+str(iteration)+'_Mar10.fits'          
      else:
          #mfile = mapdir+'Zw3146_Minkasi_Struct_Map_Pass5_Feb17.fits'
          #mfile = mapdir+'Zw3146_Minkasi_SVD10_Struct_Map_Pass10_Mar10.fits'
          mfile = mapdir+'Zw3146_Minkasi_SVD10_Rings_Struct_Map_Pass'+str(iteration)+'_Mar10.fits'

    return mfile

def find_centroid(cluster,reduc='IDL',myiter=1,oned=True):

    if reduc == 'Minkasi' or reduc == 'IDL':
        img,hdr,wtmap,wthdr = get_maps_clus(cluster,reduc=reduc,myiter=myiter,orig=True)
        
    ras, decs, pixs = astro_from_hdr(hdr)
    xymap     = get_xymap(img,pixs,0,0,oned=oned)
    x,y = xymap
    wcscen = get_centroid(cluster)
    w = wcs.WCS(hdr)
    x0,y0 =  w.wcs_world2pix(wcscen[0].to('degree'),wcscen[1].to('degree'),cpix)
    xstd = 25.0;    ystd = 20.0
    theta = 0.5
    mnlvl = 4.0e-6; myamp = -3.0e-4

    initial_guess = [myamp,x0,y0,xstd,ystd,theta,mnlvl]
    pflat = img.flatten()

    r = np.sqrt( (xymap[0] - x0)**2 + (xymap[1] - y0)**2 )
    besti = (r < 180.0)
    popt, pcov = opt.curve_fit(my2dGauss, (x[besti], y[besti]), pflat[besti], p0=initial_guess)
    print(popt)
    myx = (popt[1] - np.min(x))/pixs; myy = (popt[2] - np.min(y))/pixs
    myra,mydec = w.wcs_pix2world(myx,myy,0)
    print(myra,mydec)
    import pdb;pdb.set_trace()

    
def add_herschel_locs(mywcs,ax,wl=500, mew=2, ms=10):

    h250l = '/home/data/Herschel/Zw3146/z3146regrid_250_micron.txt'
    h350l = '/home/data/Herschel/Zw3146/z3146regrid_350_micron.txt'
    h500l = '/home/data/Herschel/Zw3146/z3146regrid_500_micron.txt'

    if wl == 500:
        hcens = get_ian_cen(h500l)
    if wl == 350:
        hcens = get_ian_cen(h350l)
    if wl == 250:
        hcens = get_ian_cen(h250l)

    hx,hy = x_centroids(hcens,mywcs,mycolor='b',mew=mew,ms=ms,ax=ax)
        
def get_ian_cen(file):

    #file='/home/data/Herschel/Zw3146/z3146regrid_500_micron.txt'
    #dtypes = ['string','float']*8
    #dtypes.append('string')
    foo = np.loadtxt(file,usecols=(1,3,5,7,9,11))
    ras  = (foo[:,0] + foo[:,1]/60.0 + foo[:,2]/3600.0)*15.0 * u.degree
    decs = (foo[:,3] + foo[:,4]/60.0 + foo[:,5]/3600.0) * u.degree

    mycens = [(ra,dec) for ra,dec in zip(ras,decs)]
    
    return mycens
