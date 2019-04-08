import numpy as np                      # A useful package...
import emcee, emcee_stats,time,datetime,os
import cProfile, sys, pstats,scipy          # I may not need pstats
import get_data_info as gdi           # Not much of a joke to make here.
import PowerLawBin as PLB
import astropy.units as u             # U just got imported!
import cr_mcmc as rcmcmc              # My version of an MCMC routine
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import PlotFittedProfile as PFP
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import Angle
import image_filtering as imf

def load_rings(option='M2',cluster='Zw3146',session=0,iteration=0):

    npydir = '/home/romero/Python/StandAlone/Comptony_Modelling/'
    sesstr=str(session)
    itestr=str(iteration)
    geom      = [0,0,0,1,1,1,0,0]  # Spherical...base assumption for all.

    if cluster == 'Zw3146':
        #nsrc    = 8
        nsrc    = 6

        if option == 'M2_ACT':
            curve = np.load(npydir+'zwicky_3146_rings_curve_act_2beam.npy')
            deriv = np.load(npydir+'zwicky_3146_rings_deriv_act_2beam.npy')
            edges = np.load(npydir+'zwicky_3146_rings_edges_act_2beam.npy')
            trim  = 5

        if option == 'M2':

            #curve = np.load(npydir+'zwicky_3146_rings_curve.npy')
            #deriv = np.load(npydir+'zwicky_3146_rings_deriv.npy')
            #edges = np.asarray([0,5,15,25,35,45,55,70,85,100,120,140,160,180,200,220,240,320,400,480],dtype='double')
            #trim  = 2
            datadir='/home/data/MUSTANG2/AGBT17_Products/Zw3146/Minkasi/'
            ######################################################################################
            ### Try without session 6:
            #curve = np.load(datadir+'zwicky_3146_rings_curve_Jan24_no_session6_TS_v0_51_Jan24_PdoCals.npy')
            #deriv = np.load(datadir+'zwicky_3146_rings_deriv_Jan24_no_session6_TS_v0_51_Jan24_PdoCals.npy')
            #edges = np.load(datadir+'zwicky_3146_rings_edges_Jan24_no_session6_TS_v0_51_Jan24_PdoCals.npy')

            if session == 0:
                #curve = np.load(datadir+'Slices/zwicky_3146_rings_curve_all_slices_rot0.0000.npy')
                #deriv = np.load(datadir+'Slices/zwicky_3146_rings_deriv_all_slices_rot0.0000.npy')
                #edges = np.load(datadir+'Slices/zwicky_3146_rings_edges_all_slices_rot0.0000.npy')
                #curve = np.load(datadir+'Slices/AndIter/zwicky_3146_rings_curve_all_slices_rot0.0000_pass_1.npy')
                #deriv = np.load(datadir+'Slices/AndIter/zwicky_3146_rings_deriv_all_slices_rot0.0000_pass_1.npy')
                #edges = np.load(datadir+'Slices/AndIter/zwicky_3146_rings_edges_all_slices_rot0.0000_pass_1.npy')

                thisdir=datadir+'IterRings/'
                #midstr='_Corr_slope_manypass_TS_v0_51_Jan24_PdoCals_svd10_pass_' # 11.1 FWHM -> slightly too big.
                #midstr='_Corr_slope_10p7_TS_v0_51_Jan24_PdoCals_svd10_pass_'  # Best overall so far??
                #midstr='_Corr_slope_10p7_TS_v0_51_Jan24_PdoCals_svd10_cmsubbed_pass_' # Has CM subbed -> filters signal
                #midstr='_Corr_slope_TS_v0_51_Jan24_PdoCals_svd10_pass_'   # OK, but SVD10 converges better

                #midstr='_Corr_slope_10p7_10c_TS_v0_51_Jan24_PdoCals_svd10_pass_' # inner rings goes to 10"
                #midstr='_Corr_slope_10p7_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_' # old edges, but all pt. srcs have 10.7 FWHM
                #midstr='_CSlopeBeam_10p7_TS_v0_51_Jan24_PdoCals_svd10_pass_'
                #midstr='_CSlopeBeam_10p7_10c_TS_v0_51_Jan24_PdoCals_svd10_pass_'
                #midstr='_CSlopeBeam_10p7_10c_TS_v0_51_Jan24_PdoCals_svd10_pass_'
                #midstr='_CSlopeBeam_10p7_XMMc_TS_v0_51_Jan24_PdoCals_svd10_pass_'
                #midstr='_CSlopeBeam_10p7_8srcs_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_'
                #########################################################################################################
                # Spherical model
                #midstr='_CSlopeBeam_10p7_6srcs_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_'  # I think my best one so far...?
                midstr='_CSlopeBeam_10p7_2019-04-06_6_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_'
                #########################################################################################################
                #midstr='_CSlopeBeam_10p7_ellipt_6_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_'
                #geom      = [0,0,0.8,1,0.8,0.8944272,0,0]  # If elliptical
                #########################################################################################################
                #midstr='_CSlopeBeam_10p7_ellipt_6_SZ2Dc_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_'
                #geom      = [-1.96,-0.26,0.8,1,0.8,0.8944272,0,0]  # If elliptical, SZ2D
                
                curve = np.load(thisdir+'zwicky_3146_rings_curve'+midstr+itestr+'.npy')
                deriv = np.load(thisdir+'zwicky_3146_rings_deriv'+midstr+itestr+'.npy')
                edges = np.load(thisdir+'zwicky_3146_rings_edges'+midstr+itestr+'.npy')
                trim  = 6
                #trim  = 8  # When working with _8srcs_
            ######################################################################################
            else:
                thisdir=datadir+'IterRings/'
                #curve = np.load(thisdir+'zwicky_3146_rings_curve_Corr_slope_manypass_TS_v0_51_Jan24_PdoCals_svd10_pass_'+sesstr+'.npy')
                #deriv = np.load(thisdir+'zwicky_3146_rings_deriv_Corr_slope_manypass_TS_v0_51_Jan24_PdoCals_svd10_pass_'+sesstr+'.npy')
                #edges = np.load(thisdir+'zwicky_3146_rings_edges_Corr_slope_manypass_TS_v0_51_Jan24_PdoCals_svd10_pass_'+sesstr+'.npy')

                midstr='_TS_v0_51_Jan24_PdoCals_svd10-Corr_slope_by_proj_with_bad_scans-0'
                curve = np.load(thisdir+'zwicky_3146_rings_curve'+midstr+sesstr+'_pass_'+itestr+'.npy')
                deriv = np.load(thisdir+'zwicky_3146_rings_deriv'+midstr+sesstr+'_pass_'+itestr+'.npy')
                edges = np.load(thisdir+'zwicky_3146_rings_edges'+midstr+sesstr+'_pass_'+itestr+'.npy')

                trim  = 6
                
    if cluster == 'HSC_2':

        datadir = '/home/data/MUSTANG2/AGBT17_Products/HSC/'
        curve   = np.load(datadir+'HSC_2_rings_curve_cow_v1.npy')
        deriv   = np.load(datadir+'HSC_2_rings_deriv_cow_v1.npy')
        edges   = np.load(datadir+'HSC_2_rings_edges_cow_v1.npy')
        trim    = 3
        nsrc    = 1
                
    if cluster == 'MOO_0105':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/'
        curve   = np.load(datadir+'MOO_0105_rings_curve_all_v5.npy')
        deriv   = np.load(datadir+'MOO_0105_rings_deriv_all_v5.npy')
        edges   = np.load(datadir+'MOO_0105_rings_edges_all_v5.npy')
        trim    = 6

    if cluster == 'MOO_0135':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/'
        curve   = np.load(datadir+'MOO_0135_rings_curve_all_Jan27.npy')
        deriv   = np.load(datadir+'MOO_0135_rings_deriv_all_Jan27.npy')
        edges   = np.load(datadir+'MOO_0135_rings_edges_all_Jan27.npy')
        trim    = 3
        nsrc    = 1

    if cluster == 'MOO_1014':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/'
        curve   = np.load(datadir+'MOO_1014_rings_curve_all_v2.npy')
        deriv   = np.load(datadir+'MOO_1014_rings_deriv_all_v2.npy')
        edges   = np.load(datadir+'MOO_1014_rings_edges_all_v2.npy')
        trim    = 3
        
    if cluster == 'MOO_1046':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/IterRings/'
        midstr  = '_Corr_slope_10p7_TS_v0_51_31_Jan_2019_svd10_pass_'
        #curve   = np.load(datadir+'MOO_1046_rings_curve_all_Jan31_0f02.npy')
        #deriv   = np.load(datadir+'MOO_1046_rings_deriv_all_Jan31_0f02.npy')
        #edges   = np.load(datadir+'MOO_1046_rings_edges_all_Jan31_0f02.npy')
        curve = np.load(datadir+'MOO_1046_rings_curve'+midstr+itestr+'.npy')
        deriv = np.load(datadir+'MOO_1046_rings_deriv'+midstr+itestr+'.npy')
        edges = np.load(datadir+'MOO_1046_rings_edges'+midstr+itestr+'.npy')
        trim    = 1
        nsrc    = 1

    if cluster == 'MOO_1059':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/IterRings/'
        midstr  = '_Corr_slope_byJy2K_TS_v0_51_31_Jan_2019_svd20_pass_'
        curve   = np.load(datadir+'MOO_1059_rings_curve'+midstr+itestr+'.npy')
        deriv   = np.load(datadir+'MOO_1059_rings_deriv'+midstr+itestr+'.npy')
        edges   = np.load(datadir+'MOO_1059_rings_edges'+midstr+itestr+'.npy')
        trim    = 1
        nsrc    = 1
        
    if cluster == 'MOO_1108':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/IterRings/'
        #midstr  = '_Corr_slope_10p7_TS_v0_51_31_Jan_2019_svd10_pass_'
        #curve   = np.load(datadir+'MOO_1110_rings_curve_all_Jan27.npy')
        #deriv   = np.load(datadir+'MOO_1110_rings_deriv_all_Jan27.npy')
        #edges   = np.load(datadir+'MOO_1110_rings_edges_all_Jan27.npy')
        curve = np.load(datadir+'MOO_1108_rings_curve'+midstr+itestr+'.npy')
        deriv = np.load(datadir+'MOO_1108_rings_deriv'+midstr+itestr+'.npy')
        edges = np.load(datadir+'MOO_1108_rings_edges'+midstr+itestr+'.npy')
        trim    = 1
        nsrc    = 1
        
    if cluster == 'MOO_1110':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/IterRings/'
        #midstr  = '_Corr_slope_10p7_TS_v0_0f02_51_Jan26_svd10_pass_'
        midstr  = '_Corr_slope_byJy2K_TS_v0_0f02_51_Jan26_svd20_pass_'
        #curve   = np.load(datadir+'MOO_1110_rings_curve_all_Jan27.npy')
        #deriv   = np.load(datadir+'MOO_1110_rings_deriv_all_Jan27.npy')
        #edges   = np.load(datadir+'MOO_1110_rings_edges_all_Jan27.npy')
        curve = np.load(datadir+'MOO_1110_rings_curve'+midstr+itestr+'.npy')
        deriv = np.load(datadir+'MOO_1110_rings_deriv'+midstr+itestr+'.npy')
        edges = np.load(datadir+'MOO_1110_rings_edges'+midstr+itestr+'.npy')
        trim    = 1
        nsrc    = 1
        
    if cluster == 'MOO_1142':

        datadir = '/home/data/MUSTANG2/AGBT18_Products/MadCOWs/Minkasi/IterRings/'
        midstr  = '_Corr_slope_10p7_TS_-20_def_ninkasi_PdoCals_svd30_pass_'
        #curve   = np.load(datadir+'MOO_1142_rings_curve_all.npy')
        #deriv   = np.load(datadir+'MOO_1142_rings_deriv_all.npy')
        #edges   = np.load(datadir+'MOO_1142_rings_edges_all.npy')
        curve = np.load(datadir+'MOO_1142_rings_curve'+midstr+itestr+'.npy')
        deriv = np.load(datadir+'MOO_1142_rings_deriv'+midstr+itestr+'.npy')
        edges = np.load(datadir+'MOO_1142_rings_edges'+midstr+itestr+'.npy')
        nsrc    = 1
        trim    = 1

    if cluster == 'MACS0717':

        thisdir='/home/data/MUSTANG2/AGBT17_Products/MACS0717/Minkasi/IterRings/'
        nsrc=3
        #midstr='_CSlopeBeam_10p7_ellipt_6_SZ2Dc_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_'
        midstr='_CSlopeBeam_10p7_3_TS_v0_51_16_Feb_2019_svd10_rfwhm_pass_'
        #geom      = [-1.96,-0.26,0.8,1,0.8,0.8944272,0,0]  # If elliptical, SZ2D
        #geom      = [0.0,0.0,0.0,1.0,1.0,1.0,0,0]  # If spherical

        curve = np.load(thisdir+'M0717_rings_curve'+midstr+itestr+'.npy')
        deriv = np.load(thisdir+'M0717_rings_deriv'+midstr+itestr+'.npy')
        edges = np.load(thisdir+'M0717_rings_edges'+midstr+itestr+'.npy')
        trim  = 3

    if cluster == 'MACS1149':

        nsrc=1
        thisdir='/home/data/MUSTANG2/AGBT17_Products/MACS1149/Minkasi/IterRings/'
        #midstr='_CSlopeBeam_10p7_ellipt_6_SZ2Dc_TS_v0_51_Jan24_PdoCals_svd10_rfwhm_pass_'
        midstr='_CSlopeBeam_10p7_1_TS_EaCMS_51_13_Mar_2019_svd10_rfwhm_pass_'
        #geom      = [-1.96,-0.26,0.8,1,0.8,0.8944272,0,0]  # If elliptical, SZ2D
        #geom      = [0.0,0.0,0.0,1.0,1.0,1.0,0,0]  # If spherical

        curve = np.load(thisdir+'MACS1149_rings_curve'+midstr+itestr+'.npy')
        deriv = np.load(thisdir+'MACS1149_rings_deriv'+midstr+itestr+'.npy')
        edges = np.load(thisdir+'MACS1149_rings_edges'+midstr+itestr+'.npy')
        trim  = 1


    print 'The shapes of curve, deriv, and edges are: ',curve.shape, deriv.shape, edges.shape
        
    return curve,deriv,edges,trim,nsrc,geom

def get_data_cov_edges(incurve,inderiv,edges,trim=0,autotrim=False,slices=False,nslice=4,slicenum=0,
                       nsrc=6,fpause=True,ptsrcamps=False):

    #import pdb;pdb.set_trace()
    if np.linalg.cond(incurve) < (1.0/sys.float_info.epsilon)**2:

        print 'Matrix is invertible. Happy day.'
       
        cov    = np.linalg.inv(incurve)
        data   = np.matmul(cov,inderiv)
        curve  = incurve

        if ptsrcamps:
            data=data[-nsrc:]
            print('Your point source amplitudes are: \n',data)
            return data
        else:
            
            if trim < 0:
                nsrc = len(inderiv) - len(edges) + 1
                trim = nsrc+1

            if slices:
                mydata = data[:-nsrc]
                mycurv = curve[:-nsrc,:-nsrc]
                data   = mydata[slicenum-1::nslice]
                curve  = mycurv[slicenum-1::nslice,slicenum-1::nslice]
                #data   = data[:nsrc-trim]
                #curve  = curve[:nsrc-trim,:nsrc-trim]
            else:
                if trim == 0:
                    print 'You selected to trim nothing, perhaps you are just testing things.'
                    #import pdb;pdb.set_trace()
            
            if trim > 0:
                data   = data[:-trim]
                curve  = curve[:-trim,:-trim]

       
    else:   
        print 'Matrix is NOT invertible. Bummer.'
        if trim > 0:
            curve = incurve[:nsrc-trim,:nsrc-trim]
            deriv = inderiv[:nsrc-trim]
        else:
            curve = incurve
            deriv = inderiv
                
        import pdb;pdb.set_trace()
        
        cov  = np.linalg.inv(curve)
        data = np.matmul(cov,deriv)

        
    ####### OK, so sometimes the data is totally unconstrained (natural), and we just need to trim stuff.
    ####### I don't know if my method is good (efficient programming) or not...

    print data      

    if fpause:
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.show()

    #import pdb;pdb.set_trace()
    
    if autotrim == True:
        gd   = (data < 0)  # For data in surface brightness units... at < 220 GHz
        gdata = data[gd]
        #myind = np.arange(len(data))[gd]
        #cmask = np.ma.make_mask(curve)
        cmask = np.ones(curve.shape,dtype='bool')
        for j in range(len(data)):
            if gd[j] == False:
                cmask[:,j]=0
                cmask[j,:]=0
                
        gcurv = curve[cmask].reshape(len(gdata),len(gdata))
        gedge = edges[:len(gdata)+1]
    else:
        gdata = data
        gcurv = curve
        gedge = edges[:len(gdata)+1]

        
    ####### Done trimming stuff.
    
    sign1 = np.matmul(gcurv,gdata)
    sign  = np.matmul(np.transpose(gdata),sign1)
    sigma = np.sqrt(sign)
    print 'Your assumed data is: ',gdata
    print 'This is over the edges: ',gedge
    print 'This data is suggesting a ',sigma,' sigma detection of the cluster.'
    #import pdb;pdb.set_trace()
    
    return gdata,gcurv,gedge

def get_start_vals(mnlvl=1.0e-5,outdir=None,cluster='Zw3146',model='NP'):

    sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)
    if type(outdir) != type(None):
        Pdlfilename = outdir+'MCMC_parameter_to_Pressure_Conversion.npy'
        np.save(Pdlfilename,Pdl2y.value)
        #mvfn = outdir+cluster+'_Map_Variables_Structured_Array.npy'
        #np.save(mvfn,map_vars)
        
    pos = PLB.a10_pres_from_rads(bins)
    if mnlvl != 0:
        pos = np.append(pos,mnlvl)        # Add a mean level to the initial values.
    #pos = list(pos)
    if model == 'GNFW':
        #pos  = np.array([1.177,8.403,5.4905,0.3081,1.0510,mnlvl])
        #pos  = np.array([8.403,mnlvl])
        pos  = np.array([1.177,8.403,mnlvl])
        bins = map_vars['thetas'] # These can just correspond to n_data points for the gNFW profiles
    if model == 'Beta':
        pos = np.asarray([pos[0]/Pdl2y.value,(map_vars['r500']/10.0).to('kpc').value,1.0,mnlvl])
        bins = map_vars['thetas']

        
    return pos,bins

def get_model(pos,ppbins,edges,inst='MUSTANG2',mytype='NP',mycluster='Default',slopes=None,ySph=False,
              geom=[0,0,0,1,1,1,0,0]):

    yProf,alphas,yint = PLB.prof_from_pars_rads(pos,ppbins,retall=True,model=mytype,cluster=mycluster,ySph=ySph,geom=geom)
    modProf = PLB.resample_prof(yProf,edges,inst=inst,slopes=slopes)
    #import pdb;pdb.set_trace()

    return modProf,alphas,yint

def make_model_map(pos,ppbins,edges,map_vars,mapshape,inst='MUSTANG2',mytype='NP',mycluster='Default',slopes=None,
                   ySph=False,xymap=None,geom=[0,0,0,1,1,1,0,0],fwhm=10.7,pixsize=2.0):

    yProf,alphas,yint = PLB.prof_from_pars_rads(pos,ppbins,retall=True,model=mytype,cluster=mycluster,ySph=ySph,geom=geom)
    tSZc,kSZc = PLB.get_conv_factors(instrument=inst)
    skymap    = grid_profile(map_vars["thetas"], yProf*tSZc, xymap, geoparams=geom)
    sig2fwhm  = np.sqrt(8.0*np.log(2.0)) 
    pix_sigma = fwhm/(pixsize*sig2fwhm)
    skymap    = skymap.reshape(mapshape)
    mymap     = scipy.ndimage.filters.gaussian_filter(skymap, pix_sigma)
    #mymap     = mymap.ravel()
    
    return mymap.ravel()

def loop_fit_profs(dataset='M2',version='-SVD10_Apr8_ySph_CSB_6srcs_sph-',cluster='Zw3146',model='NP',domnlvl=True,doemcee=False,session=0,
                   nslice=0,ring_combine=False,slope=True,slices=False,npass=10,nsession=0,sstart=0,istart=1,ySph=False,
                   longrun=False,rfwhm=10.7):

    for jj in range(sstart,nsession+1):
        for kk in range(istart,npass+1):
            fit_prof_to_rings(dataset=dataset,version=version,cluster=cluster,model=model,domnlvl=True,doemcee=False,session=jj,
                              nslice=0,slicenum=0,ring_combine=False,slope=slope,slices=False,fpause=False,makemap=True,
                              iteration=kk,ySph=ySph,longrun=longrun,rfwhm=rfwhm)


def fit_prof_to_rings(dataset='M2',version='-Apr8_rfwhm_6srcs_ell-',cluster='Zw3146',model='NP',domnlvl=True,doemcee=False,session=0,
                      nslice=8,slicenum=0,ring_combine=False,slope=True,slices=False,fpause=True,makemap=True,iteration=1,ySph=False,
                      longrun=False,rfwhm=10.7):

    """
    Purpose: This is the workhorse module for fitting pressure profiles to surface brightness profiles AND producing ancillary high-level products/results.

    Inputs:
    ---> dataset:  A string that defines which instrument the data comes from. For now, we really only use MUSTANG-2 ('M2') data.
    ---> version:  A string that identifies parameters used in your fitting procedure.
    ---> cluster:  A string that identifies the galaxy cluster for which the data pertains. This finds necessary priors (redshift and electron temperature) that are necessary for determining pressure profiles.
    ---> model:    Either of: "NP", for Non-parametric, "GNFW" for Generalized NFW, or "BETA" for a beta profile.
    ---> domnlvl:  Do you want to fit for a mean level? (In general, you should)
    ---> doemcee:  Use emcee or a custom (single chain) MCMC?
    ---> session:  If you are looking to load rings from different observation sessions (nights), you can set this keyword to the appropriate number (only used if this is greater than 0).
    ---> nslice:   Number of slices used if you have rings by the slice.
    ---> slicenum: So to something greater than 0 if you are analyzing sliced profiles.
    ---> ring_combine: If you want to combine slices and fit them as if they were complete rings, set this to True.
    ---> slope:    Do the annuli use slopes? (Nowadays the default is Yes, i.e. True)
    ---> fpause:   Force pause? Good if you want to check things, but if looping, better to not have pauses.
    ---> makemap:  Map a residual map (and add models as other extensions). Good to do - can check and use for other things.
    ---> iteration: The iteration that you want to analyze.
    ---> ySph:     Do you want to use yCyl or ySph? The difference appears to be negligible.
    ---> longrun:  The default length usually converges, but the "burn in" may not be quite long enough. Use longrun to ensure convergence / good parameter covariances.
    ---> rfwhm:    Restricted FWHM: If the point sources have a FWHM < rfwhm, then use rfwhm instead. This is used because sometimes Minkasi fits a beam that is ~< 8", but we know that M2 has a broader beam than that - especially over all nights.


    """

    version = model+version
    ### You might want "_Iter" to be "_Session" depending on what you are actually doing...
    outdir = '/home/romero/Results_Python/Rings/'+cluster+'/'
    if slicenum > 0:
        outdir = outdir+'Slices/'
    if session > 0:
        version = version+'Session'+str(session)+'-'
        outdir = outdir+'BySession/'
    if iteration > 0:
        outdir = outdir+'Iter/'
        version = version+'Iter'+str(iteration)
        
    if slicenum > 0:
        #session=slicenum
        version = version+'Slice'+str(slicenum)
        slices=True
    #if nslice > 0:
    #    version = version+'Slice'+str(slicenum+1)
    #    slices=True
    ### dataset can be either 'M2' or 'M2_ACT'

    mymnlvl=1.0e-5 if domnlvl else 0.0

    slopes=None
    if ring_combine:
        gdata,gcurv,gedge      = combine_rings(version=version,cluster=cluster,nslice=nslice)
    else:
        curve,deriv,edges,trim,nsrc,geom = load_rings(option=dataset,cluster=cluster,session=session,iteration=iteration)
        gdata,gcurv,gedge      = get_data_cov_edges(curve,deriv,edges,trim=trim,slices=slices,nslice=nslice,slicenum=slicenum,
                                                    nsrc=6,fpause=fpause)
        if slope:
            #gdata,gcurv,gedge      = get_better_slope_rings(option=dataset,cluster=cluster,session=session)
            slopes = get_slope(option=dataset,cluster=cluster,mypass=iteration,slices=slices,gedge=gedge,trim=trim,
                               nsrc=nsrc,fpause=fpause,session=session)
            print 'Your slopes are: ',slopes
            print '=============================================================='
        #else:
        #    curve,deriv,edges,trim,nsrc,geom= load_rings(option=dataset,cluster=cluster,session=session)
        #    gdata,gcurv,gedge      = get_data_cov_edges(curve,deriv,edges,trim=trim,slices=slices,nslice=nslice,slicenum=slicenum,nsrc=6)
    pos,bins          = get_start_vals(mnlvl=mymnlvl,outdir=outdir,cluster=cluster,model=model)
    startval = pos.copy()     #pos*1.0

    nchain=1

    ### This is a bit paternalistic...but I don't have the covariances for other things yet.
    if not (cluster == 'Zw3146' and model == 'NP' and len(bins) == 6):
        doemcee = True
    
    if doemcee == True:
        nchain= 2*(len(pos)+4) 
        print(nchain)
        #import pdb;pdb.set_trace()
    
    ndim = len(pos); nsteps = 100000/nchain
    if model == 'GNFW':
        nsteps /= 4
    if longrun:
        nsteps*= 3
        version = version+'_long'
    #nburn = nsteps*nchain/200
    nburn = nsteps/20

    pr = cProfile.Profile()
    pr.enable()

    if doemcee == True:
        sampler, t_mcmc, ConvTests = run_emcee(gdata,gcurv,pos,bins,gedge,nsteps=nsteps,nwalkers=nchain,ndim=ndim,nburn=nburn,
                                               mytype=model,domnlvl=domnlvl,mycluster=cluster,slopes=slopes,ySph=ySph)
    else:
        sampler, t_mcmc, ConvTests = run_mymcmc(gdata,gcurv,pos,bins,gedge,nsteps=nsteps,nwalkers=nchain,ndim=ndim,nburn=nburn,
                                                mytype=model,domnlvl=domnlvl,mycluster=cluster,slopes=slopes,ySph=ySph)
        
    pr.disable()
    #########################################################################
    prename = dataset+'_'+cluster+'_'+version+'_'
    prof_out = outdir+'PythonProfilerResults_crmcmc.txt'
    sys.stdout = open(prof_out, 'w')
    pr.print_stats()
    sys.stdout = sys.__stdout__

    
    blobarr     = np.array(sampler.blobs)
    stepname = prename+'_steps_when_fitting_to_rings.png'
    PFP.plot_steps(sampler,outdir,stepname,burn_in=nburn)

    samples    = sampler.chain[:,nburn:, :].reshape((-1,ndim))
    solns      = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0))))
    goodY      = blobarr[nburn:,:,:].reshape((-1,1))
    #lessmnlvl  = solns[:-1,:]
    mybest     = solns[:,0]
    print "Best results - without the mean level ",mybest
    #import pdb;pdb.set_trace()
    bmodel,boutalphas,byint = get_model(mybest,bins,gedge,mytype=model,mycluster=cluster,slopes=slopes,ySph=ySph)
    imodel,ioutalphas,iyint = get_model(startval,bins,gedge,mytype=model,mycluster=cluster,slopes=slopes,ySph=ySph)
    #pbins = bins
    #if model == 'Beta' or model == 'gNFW': pbins = map_vars['thetas']*180.0*3600.0/np.pi
    
    iYsave = outdir+prename+'IntegratedYsphs.npy' if ySph else outdir+prename+'IntegratedYcyls.npy'
    np.save(iYsave,goodY)
    np.save(outdir+prename+'Samples.npy',samples)
    np.save(outdir+prename+'Solutions.npy',solns)
    np.save(outdir+prename+'Chain.npy',sampler.chain)

    #import pdb;pdb.set_trace()

    sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)
    if model == 'NP':
        arcsecs = bins[...,np.newaxis] * (u.rad).to('arcsec')
        radsANDpres = np.hstack((arcsecs,solns[:-1,:]/Pdl2y))
        pamphdr = 'Radius (arcsec), Pressure (keV cm**-3), Upper Uncertainty, Lower Uncertainty'
        np.savetxt(outdir+prename+'Pressure_Amplitudes.txt',radsANDpres,header=pamphdr)
    #np.save(outdir+'SamplerObject.npy',sampler,allow_pickle=True)

    #import pdb;pdb.set_trace()

    #stepname = prename+'_steps_when_fitting_to_rings.png'
    presname = prename+'_fitted_pressure_profile_A10.png'
    #PFP.plot_steps(sampler,outdir,stepname,burn_in=nburn)
    PFP.plot_pres_bins(solns,dataset,outdir,presname,cluster=cluster,
                       IntegratedYs=goodY,overlay='a10',mymodel=model,ySph=ySph,geom=geom)
    barename = prename+'_fitted_pressure_profile_bare.png'
    PFP.plot_pres_bins(solns,dataset,outdir,barename,cluster=cluster,IntegratedYs=goodY,overlay='None',mymodel=model,bare=True,ySph=ySph,geom=geom)
    profname = prename+'_fitted_brightness_profile.png'
    PFP.plot_surface_profs(bmodel,bins,gdata,gcurv,gedge,outdir,profname,cluster=cluster,mymodel=model,pinit=imodel,slopes=slopes)
    profname = prename+'_fitted_brightness_profile_bare.png'
    PFP.plot_surface_profs(bmodel,bins,gdata,gcurv,gedge,outdir,profname,cluster=cluster,mymodel=model,pinit=imodel,bare=True,slopes=slopes)
    if cluster == 'Zw3146':
        presname = prename+'_fitted_pressure_profile_XMM.png'
        PFP.plot_pres_bins(solns,dataset,outdir,presname,cluster=cluster,
                           IntegratedYs=goodY,overlay='XMM',mymodel=model,ySph=ySph,geom=geom)

    corrname = dataset+"_correlations_via_corner"+version+".png"

    PFP.plot_correlations(samples,outdir,corrname,blobs=goodY,mtype=model,domnlvl=domnlvl)
    #PFP.plot_ConvTests(ConvTests,outdir,mpp=8,name=cluster)
    find_cov_mcmc_pars(samples,outdir,prename)
    PFP.plot_autocorrs(sampler,outdir,prename+'Autocorrelations.png',burn_in=200)

    if makemap:
        ptamps = get_data_cov_edges(curve,deriv,edges,trim=trim,slices=slices,nslice=nslice,slicenum=slicenum,
                                                    nsrc=nsrc,fpause=fpause,ptsrcamps=True)
        print(ptamps)
        make_fits_from_fits(cluster,pos=mybest,ppbins=bins,edges=gedge,version=version,dataset=dataset,
                            inst='MUSTANG2',mytype=model,ptamps=ptamps,iteration=iteration,rfwhm=rfwhm,geom=geom)
    
    if fpause:
        import pdb;pdb.set_trace()
    #PFP.plot_ConvTests(ConvTests,outdir,mpp=8,name=cluster)
    #beep = sampler.chain[0,:,0]
    #boop = emcee_stats.autocorr_func_1d(beep)

    
    #return sampler,ConvTests

def find_cov_mcmc_pars(samples,outdir,prename):

    ### Which axis?
    msubbed = samples - np.mean(samples,axis=0)
    sashape = samples.shape
    cov = np.matmul(np.transpose(msubbed),msubbed) / sashape[0]

    fullname = outdir+prename+'Parameter_Covariance.npy'
    np.save(fullname,cov)

def replot_with_XMM():

    import pdb;pdb.set_trace()
    PFP.plot_pres_bins(solns,dataset,outdir,presname,cluster=cluster,
                       IntegratedYs=goodY,overlay='a10',mymodel=model)

################################################################################

################################################################################

################################################################################

################################################################################
    
def run_emcee(data,curve,myargs,ppbins,edges,nsteps=2500,nwalkers=20,ndim=1,nburn=200,mytype='NP',
              domnlvl=True,mycluster='Default',slopes=None,ySph=False):

    if ndim < len(myargs): ndim  = len(myargs)
    if nburn < nsteps/10: nburn = nsteps/10
    print('N dimensions: ',ndim,'N burn: ',nburn,'N steps: ',nsteps,'N walkers: ',nwalkers)
    
    def lnlike(pos):                          ### emcee_fitting_vars

        #import pdb;pdb.set_trace()
        model,outalphas,yint = get_model(pos,ppbins,edges,mytype=mytype,mycluster=mycluster,slopes=slopes,ySph=ySph)

        #mnlvl=0
        mnlvl=pos[-1] if domnlvl else 0.0

        diff   = data - model -mnlvl
        interm = np.matmul(curve,diff)
        chisq  = np.matmul(np.transpose(diff),interm)
        loglike = -0.5 * chisq
        #loglike-= 0.5 * (np.sum(((model - data)**2) * weights))
        ycyl = [yint]
        
        return loglike,outalphas,ycyl
    
    def lnprior(pos,outalphas,ycyl):
        
        plike = 0.0
        addlike=0.0

        slopeok = False if outalphas[-1] <= 1.0 else True
        # # # # # # # # # # # # # # # # # # # # # # # # #          
        #if len(priorunc) > 0:
        #    addlike =  -0.5* (np.sum(((pwithpri - mypriors)**2) / priorunc**2))

        #for myouts in outalphas:
        #    if len(myouts) == 0:
        #        slopeok = True
        #    else:
        #        if myouts[-1] > 1.0: slopeok = True
        #        if myouts[-1] <= 1.0:
        #            slopeok = False
        #            break
        #    #print(slopeok)

        ### Get the pressure profile parameters (without a mean level)
        prespos = pos[0:-1] # Define as separate variable here... 
        if all([param > 0.0 for param in prespos]) and (slopeok == True):
            #print 'Everything OK'
            return plike+addlike
        #print 'In LnPrior, LogLike set to infinity ', outalphas[-1]
        #import pdb;pdb.set_trace()
        return -np.inf

    def lnprob(pos):
        likel,outalphas,ycyl = lnlike(pos)
        #yarr = np.array(ycyl)
        if not np.isfinite(likel):
            #print 'In LnProb, LogLike set to infinity'
            return -np.inf, [-1.0 for ybad in ycyl]
        lp = lnprior(pos,outalphas,ycyl)
        #print(lp,likel,ycyl)
        #import pdb;pdb.set_trace()
        if not np.isfinite(lp):
            #print 'In LnProb, LogLike set to infinity'
            return -np.inf , [-1.0 for ybad in ycyl]
        return lp + likel , ycyl


    t_premcmc = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim,lnprob, threads = 1)
    #import pdb;pdb.set_trace()
    
    pos = [(myargs + np.random.randn(ndim) * myargs/1e3) for i in range(nwalkers)]
    proccheck = np.max([100,nsteps/50])  # Check in every 100 steps (iterations)
    #proccheck = 100  # Check in every 100 steps (iterations)
    fts = int(np.ceil(nsteps*1.0/proccheck))
    imo = int(ndim+1)
    print('Checking every ',proccheck,' steps')
    #import pdb;pdb.set_trace()
    gw2010 = np.zeros((fts,imo))
    newmet = np.zeros((fts,imo))
    dt0 = datetime.datetime.now()
    
    for i, result in enumerate(sampler.sample(pos, iterations= nsteps)):
        #print(i)
        if (i+1) % proccheck == 0:
            cind     = int((i+1) / proccheck)-1
            for jjj in range(ndim):
                #dcheck = sampler.chain.shape
                #print dcheck
                gw2010[cind,jjj]   = emcee_stats.autocorr_gw2010(sampler.chain[:,:i+1,jjj])
                newmet[cind,jjj]   = emcee_stats.autocorr_new(sampler.chain[:,:i+1,jjj])
            gw2010[cind,-1]   = emcee_stats.autocorr_gw2010(sampler._lnprob.T)
            newmet[cind,-1]   = emcee_stats.autocorr_new(sampler._lnprob.T)
            
            t_so_far = time.time() - t_premcmc
            perdone  = float(i+1) / nsteps
            t_total  = t_so_far / perdone
            t_remain = (t_total * (1.0 - perdone) * u.s).to("min")
            t_finish = dt0 + datetime.timedelta(seconds=t_total)
            print("{0:5.1%}".format(perdone)+' done; >>> Estimated Time Remaining: ',
                  t_remain.value,' minutes; for a finish at: ',t_finish.strftime("%Y-%m-%d %H:%M:%S"))
            #print(pos)
            #import pdb;pdb.set_trace()
            #print "Average time per step so far: ", "{:5.1f}".format(t_so_far/(i+1.0))," seconds."
            
    myabscissa = np.arange(np.ceil(nsteps*1.0/ proccheck))*proccheck
    ConvTests={'GoodmanWeare2010':gw2010,'Fardal_emcee':newmet,'Abscissa':myabscissa}
    
    #for jjj in range(hk.cfp.ndim):
    #    gw2010 = emcee_stats.autocorr_gw2010(sampler.chain[:,:,jjj])
    #    newmet = emcee_stats.autocorr_new(sampler.chain[:,:,jjj])
    #sampler.run_mcmc(pos,hk.cfp.nsteps)
    
    t_mcmc = time.time() - t_premcmc
    dt_end = datetime.datetime.now()
    esttime = 30.0
    
    print "MCMC time: ",t_mcmc/60.0,' minutes'
    print "Difference from predicted: ", (t_mcmc - esttime),' seconds'
    print "Finishing time: ", dt_end.strftime("%Y-%m-%d %H:%M:%S")
    print "This is equivalent to ",t_mcmc/nsteps," seconds per step."
    print "This is equivalent to ",1000.0*t_mcmc/nsteps/nwalkers," milliseconds per model."
    print "Initial Guesses: ", myargs

    #import pdb;pdb.set_trace()

    return sampler, t_mcmc, ConvTests

def run_mymcmc(data,curve,myargs,ppbins,edges,nsteps=2500,nwalkers=1,ndim=1,nburn=200,mytype='NP',
               domnlvl=True,mycluster='Default',slopes=None,ySph=False):

    if ndim < len(myargs): ndim  = len(myargs)
    if nburn < nsteps/10: nburn = nsteps/10
    #ringdir = '/home/romero/Results_Python/Rings/'
    ringdir = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/Minkasi/'
    if mycluster == 'Zw3146' and mytype == 'NP':
        pcovfile = ringdir+'M2_Zw3146_NP_cr4_Parameter_Covariance.npy'
    else:
        raise Exception('Unknown territory. Must provide appropriate covariance matrix to inform MC steps to take')
        
    def lnlike(pos):                          ### emcee_fitting_vars

        #import pdb;pdb.set_trace()
        model,outalphas,yint = get_model(pos,ppbins,edges,mytype=mytype,mycluster=mycluster,slopes=slopes,ySph=ySph)

        #mnlvl=0
        mnlvl=pos[-1] if domnlvl else 0.0

        diff   = data - model -mnlvl
        interm = np.matmul(curve,diff)
        chisq  = np.matmul(np.transpose(diff),interm)
        loglike = -0.5 * chisq
        #loglike-= 0.5 * (np.sum(((model - data)**2) * weights))
        ycyl = [yint]
        
        return loglike,outalphas,ycyl
    
    def lnprior(pos,outalphas,ycyl):
        
        plike = 0.0
        addlike=0.0

        slopeok = False if outalphas[-1] <= 1.0 else True
        # # # # # # # # # # # # # # # # # # # # # # # # #          
        #if len(priorunc) > 0:
        #    addlike =  -0.5* (np.sum(((pwithpri - mypriors)**2) / priorunc**2))

        #for myouts in outalphas:
        #    if len(myouts) == 0:
        #        slopeok = True
        #    else:
        #        if myouts[-1] > 1.0: slopeok = True
        #        if myouts[-1] <= 1.0:
        #            slopeok = False
        #            break
        #    #print(slopeok)

        ### Get the pressure profile parameters (without a mean level)
        prespos = pos[0:-1] # Define as separate variable here... 
        if all([param > 0.0 for param in prespos]) and (slopeok == True):
            #print 'Everything OK'
            return plike+addlike
        #print 'In LnPrior, LogLike set to infinity ', outalphas[-1]
        #import pdb;pdb.set_trace()
        return -np.inf

    def lnprob(pos):
        likel,outalphas,ycyl = lnlike(pos)
        #yarr = np.array(ycyl)
        if not np.isfinite(likel):
            #print 'In LnProb, LogLike set to infinity'
            return -np.inf, [-1.0 for ybad in ycyl]
        lp = lnprior(pos,outalphas,ycyl)
        #import pdb;pdb.set_trace()
        if not np.isfinite(lp):
            #print 'In LnProb, LogLike set to infinity'
            return -np.inf , [-1.0 for ybad in ycyl]
        return lp + likel , ycyl


    t_premcmc = time.time()
    #sampler = emcee.EnsembleSampler(nwalkers, ndim,lnprob, threads = 1)
    sampler = rcmcmc.Sampler(ndim,lnprob,nchains=nwalkers)
    mycov   = rcmcmc.get_cov(pcovfile)
    
    #import pdb;pdb.set_trace()
    #pos = myargs
    pos = [(myargs + np.random.randn(ndim) * myargs/1e3) for i in range(nwalkers)]
    proccheck = np.max([100,nsteps/10])  # Check in every 100 steps (iterations)
    corrcheck = np.min([proccheck*5,nsteps])
    #proccheck = 100  # Check in every 100 steps (iterations)
    fts = int(np.ceil(nsteps*1.0/corrcheck))
    imo = int(ndim+1)
    #import pdb;pdb.set_trace()
    gw2010 = np.zeros((fts,imo))
    newmet = np.zeros((fts,imo))
    dt0 = datetime.datetime.now()
    
    for i, result in enumerate(sampler.sample(pos, iterations= nsteps,pcov=mycov)):
    #    print i
        if (i+1) % corrcheck == 0:
            cind     = int((i+1) / corrcheck)-1
            for jjj in range(ndim):
                #dcheck = sampler.chain.shape
                #print dcheck
                gw2010[cind,jjj]   = emcee_stats.autocorr_gw2010(sampler.chain[:,:i+1,jjj])
                newmet[cind,jjj]   = emcee_stats.autocorr_new(sampler.chain[:,:i+1,jjj])
            gw2010[cind,-1]   = emcee_stats.autocorr_gw2010(sampler._lnprob.T)
            newmet[cind,-1]   = emcee_stats.autocorr_new(sampler._lnprob.T)

        if (i+1) % proccheck == 0:
            t_so_far = time.time() - t_premcmc
            perdone  = float(i+1) / nsteps
            t_total  = t_so_far / perdone
            t_remain = (t_total * (1.0 - perdone) * u.s).to("min")
            t_finish = dt0 + datetime.timedelta(seconds=t_total)
            print "{0:5.1%}".format(perdone)+' done; >>> Estimated Time Remaining: ',\
                t_remain.value,' minutes; for a finish at: ',t_finish.strftime("%Y-%m-%d %H:%M:%S")
            #print "Average time per step so far: ", "{:5.1f}".format(t_so_far/(i+1.0))," seconds."
            
    myabscissa = np.arange(np.ceil(nsteps*1.0/ proccheck))*proccheck
    ConvTests={'GoodmanWeare2010':gw2010,'Fardal_emcee':newmet,'Abscissa':myabscissa}
    
    #for jjj in range(hk.cfp.ndim):
    #    gw2010 = emcee_stats.autocorr_gw2010(sampler.chain[:,:,jjj])
    #    newmet = emcee_stats.autocorr_new(sampler.chain[:,:,jjj])
    #sampler.run_mcmc(pos,hk.cfp.nsteps)
    
    t_mcmc = time.time() - t_premcmc
    dt_end = datetime.datetime.now()
    esttime = 30.0
    
    print "MCMC time: ",t_mcmc/60.0,' minutes'
    print "Difference from predicted: ", (t_mcmc - esttime),' seconds'
    print "Finishing time: ", dt_end.strftime("%Y-%m-%d %H:%M:%S")
    print "This is equivalent to ",t_mcmc/nsteps," seconds per step."
    print "This is equivalent to ",1000.0*t_mcmc/nsteps/nwalkers," milliseconds per model."
    print "Initial Guesses: ", myargs

    #import pdb;pdb.set_trace()

    return sampler, t_mcmc, ConvTests


def combine_rings(dataset='M2',version='-Feb5_IRings_v2-',cluster='Zw3146',model='NP',domnlvl=True,doemcee=False,session=0,
                  nslice=4,slicenum=0,iteration=0):

    version = model+version
    ### You might want "_Iter" to be "_Session" depending on what you are actually doing...
    outdir = '/home/romero/Results_Python/Rings/'+cluster+'/'
    if slicenum > 0:
        outdir = outdir+'Slices/'
    if session > 0:
        outdir = outdir+'Iter/'

    #nslice=8
    gpfile = gpfile_by_clus(cluster)
    gpars=np.load(gpfile)
    x    = gpars[0::4]*np.cos(gpars[1::4])
    y    = gpars[1::4]
    xx   = (x[0] - x)*3600*180/np.pi
    yy   = (y - y[0])*3600*180/np.pi
    rr   = np.sqrt(xx**2 + yy**2)
    tt   = np.arctan2(yy,xx)
    tdeg = tt*180.0/np.pi
    tsli = np.floor((tt+np.pi)*nslice/(2.0*np.pi))+1
    curve,deriv,edges,trim,nsrc,geom = load_rings(option=dataset,cluster=cluster,session=session,iteration=iteration)
    npar = ( len(edges)-1 )
    nsrc = 6
    lhs  = np.zeros((npar+nsrc,npar+nsrc))
    rhs  = np.zeros((npar+nsrc))
    nlow = npar

    fullind = range(curve.shape[0]-nsrc)
    #for i in range(nslice):
    for i in range(npar+nsrc):
        #miind=fullind[i::nslice]
        miind = np.arange(i*nslice,(i+1)*nslice)
        if i < npar:
            rhs[i] = np.sum(deriv[miind])
        else:
            rhs[i] = deriv[npar*nslice+i-npar]

        for j in range(i,npar+nsrc):
            mjind    = np.arange(j*nslice,(j+1)*nslice)
            if i < npar and j < npar:
                #tmp = curve[miind,mjind]
                tmp = curve[i*nslice:(i+1)*nslice,j*nslice:(j+1)*nslice]
                #print tmp.shape
                lhs[i,j] = np.sum(tmp)
            else:
                if i >= npar and j >= npar:
                    lhs[i,j] = np.sum(curve[npar*nslice+i-npar,npar*nslice+j-npar])  # Sum is not necessary here...
                else:
                    if i < npar:
                        #print i, j,npar
                        #import pdb;pdb.set_trace()
                        lhs[i,j] = np.sum(curve[i*nslice:(i+1)*nslice,npar*nslice+j-npar])
                        #lhs[i,j] = np.sum(curve[miind,npar*nslice+j-npar])
                        #check    = np.sum(curve[npar*nslice+j-npar,miind])
                        #print lhs[i,j],check
                    else:
                        print 'Else ',i, j,npar      # Should never get here.
                        lhs[i,j] = np.sum(curve[npar*nslice+i-npar,mjind])
            lhs[j,i] = lhs[i,j]

    print 'Done adding'
    slicenum=0
    slices=False
    gdata,gcurv,gedge  = get_data_cov_edges(lhs,rhs,edges,trim=6,slices=slices,nslice=nslice,slicenum=slicenum,nsrc=6)

    import pdb;pdb.set_trace()

    return gdata,gcurv,gedge

def get_better_slope_rings(option='M2',cluster='Zw3146',session=0,slices=False,nslice=0,slicenum=0,iteration=0):
    '''
    This should be considered defunct now. I prefer to do the weighting elsewhere.
    '''
    
    #slices=False
    mypass=session
    curve,deriv,edges,trim,nsrc,geom = load_rings(option=option,cluster=cluster,session=mypass,iteration=iteration)
    gdata,gcurv,gedge      = get_data_cov_edges(curve,deriv,edges,trim=trim,slices=slices,nslice=nslice,slicenum=slicenum,nsrc=nsrc)

    slopes = get_slope(option=option,cluster=cluster,mypass=mypass,slices=slices,nslice=nslice,slicenum=slicenum,trim=trim,nsrc=nsrc)
            
    nom = gdata* (( (gedge[1:]**2)/2 + (slopes*gedge[1:]**3)/3 ) - ( (gedge[:-1]**2)/2 + (slopes*gedge[:-1]**3)/3 ))
    den = ( (gedge[1:]**2 - gedge[:-1]**2) * 1.0/2.0)

    wted_data = nom/den
    scale = wted_data/gdata
    #import pdb;pdb.set_trace()

    return wted_data,gcurv,gedge

def get_slope(option='M2',cluster='Zw3146',mypass=0,slices=False,nslice=0,slicenum=0,gedge=[],trim=0,nsrc=0,fpause=False,session=0):

    mytrim = trim
    if mypass > 1:
        pcurve,pderiv,pedges,ptrim,pnsrc,geom = load_rings(option=option,cluster=cluster,iteration=mypass-1,session=session)
        pdata,pcurv,pedge      = get_data_cov_edges(pcurve,pderiv,pedges,trim=mytrim,slices=slices,nslice=nslice,slicenum=slicenum,
                                                    nsrc=nsrc,fpause=fpause)
        myvals = pdata - np.max(pdata)
        pk2pk  = np.max(myvals) - np.min(myvals)
        myvals-= pk2pk/50.0
        slopes = (myvals[:-1] - myvals[1:])/(pedge[1:-1]-pedge[:-2])
        if nsrc+1-trim < 0:
            slopes = slopes[:nsrc+1-trim]/myvals[:nsrc-trim]
            print('All good with the slopes array.')
        else:
            slopes = slopes/myvals[:-1]
            slopes = np.append(slopes,[0])
            print('Had to add a zero onto the slopes array.')
    else:
        slopes = np.zeros((len(gedge)-1))

    return slopes

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

def get_map_products(cluster='Zw3146',iteration=1):

    sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)

    mfile = get_map_file(cluster=cluster,SNR=False,iteration=iteration)
    data_map, header = fits.getdata(mfile, header=True)
    mfile = get_map_file(cluster=cluster,SNR=True,iteration=iteration)
    rms_map, rmshdr = fits.getdata(mfile, header=True,ext=1)
    nfile = get_map_file(cluster=cluster,Noise=True,iteration=iteration)
    noise_map, noihdr = fits.getdata(nfile, header=True,ext=0)

    ra0  = map_vars["racen"].to('deg').value
    dec0 = map_vars["deccen"].to('deg').value
    w = wcs.WCS(header)
    mypix = w.wcs_world2pix(ra0,dec0,0)  # Had been set to 1, but now I want to start at zero.
    print('----------------------------------------------------')
    print('Pixel centroid is: ',mypix)
    print('----------------------------------------------------')
    #import pdb;pdb.set_trace()
    
    ras, decs, pixs = astro_from_hdr(header)
    print 'Pixel size is: ',pixs

    #xymap = get_xymap(data_map,pixs,mypix[0],mypix[1])
    xymap = get_xymap(data_map,pixs,mypix[0],mypix[1])

    return data_map, header, rms_map, rmshdr, xymap, bins, map_vars, noise_map
    
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

def get_radial_map(map,pixsize,xcentre=[],ycentre=[]):

    x,y = get_xymap(map,pixsize,xcentre=xcentre,ycentre=ycentre)
    r = np.sqrt(x*x +y*y)

    return r

def grid_profile(rads, profile, xymap, geoparams=[0,0,0,1,1,1,0,0],myscale=1.0,axis='z'):

    ### Get new grid:
    arc2rad =  4.84813681109536e-06 # arcseconds to radians
    (x,y) = xymap
    x,y = rot_trans_grid(x,y,geoparams[0],geoparams[1],geoparams[2]) # 0.008 sec per call
    x,y = get_ell_rads(x,y,geoparams[3],geoparams[4])                # 0.001 sec per call
    theta = np.sqrt(x**2 + y**2)*arc2rad
    theta_min = rads[0]*2.0 # Maybe risky, but this is defined so that it is sorted.
    bi=(theta < theta_min);   theta[bi]=theta_min
    mymap  = np.interp(theta,rads,profile)
    
    if axis == 'x':
        xell = (x/(geoparams[3]*myscale))*arc2rad # x is initially presented in arcseconds
        modmap = geoparams[5]*(xell**2)**(geoparams[6]) # Consistent with model creation??? (26 July 2017)
    if axis == 'y':
        yell = (y/(geoparams[4]*myscale))*arc2rad # x is initially presented in arcseconds
        modmap = geoparams[5]*(yell**2)**(geoparams[6]) # Consistent with model creation??? (26 July 2017)
    if axis == 'z':
        modmap = geoparams[5]

    if modmap != 1:
        mymap *= modmap   # Very important to be precise here.
    if geoparams[7] > 0:
        angmap = np.arctan2(y,x)
        bi = (abs(angmap) > geoparams[7]/2.0)
        mymap[bi] = 0.0

    return mymap

def get_ell_rads(x,y,ella,ellb):

    xnew = x/ella ; ynew = y/ellb

    return xnew, ynew
    
def rot_trans_grid(x,y,xs,ys,rot_rad):

    ### Re-written Aug. 2018 to add at least a factor of 2 in speed... Hopefully more.
    newx  = (x - xs)
    newy  = (y - ys)
    sinr = np.sin(rot_rad)
    cosr = np.cos(rot_rad)
    xy   = np.vstack((newx,newy))
    mymat= np.array([[cosr,sinr],[-sinr,cosr]])
    xxyy = np.dot(mymat,xy)


    return xxyy[0,:],xxyy[1,:]

def make_hdu(modelmap,modelhdr,ptmap=None,datamap=None,rmsmap=None,nmap=None,cluster='Zw3146',mnlvl=0,
             smooth=10.0,fullpath=None,geom=[0,0,0,1,1,1,0,0],units='K_RJ'):

    hdu0 = fits.PrimaryHDU(modelmap,header=modelhdr)
    hdu0.header.append(("Title","Cluster Model"))
    hdu0.header.append(("Target",cluster))          
    hdu0.header.append(("Xoffset",geom[0],'X (arcsecond) from reference centroid'))          
    hdu0.header.append(("Yoffset",geom[1],'Y (arcsecond) from reference centroid'))          
    hdu0.header.append(("Rot_Ang",geom[2],'Rotation, in radians'))          
    hdu0.header.append(("Major",geom[3],'Major (X) axis scaling'))          
    hdu0.header.append(("Minor",geom[4],'Minor (Y) axis scaling'))          
    hdu0.header.append(("LOS",geom[5],'Line of sight (Z) axis scaling'))          
    hdu0.header.append(("Xi",geom[6],'Taper scaling (power law)'))          
    hdu0.header.append(("Opening",geom[7],'Opening angle (radians)'))          
    hdu0.header.append(("MnLvl",mnlvl,'Mean Level (make units), not included here.'))          
    hdu0.header.append(("Units",units,'Map Units'))          
    hdu0.name = 'Cluster_Model'
    myhdu = [hdu0]
    pixsize  = 2.0 # Need to not have this hard-coded
    sig2fwhm = np.sqrt(8.0*np.log(2.0)) 
    
    if not(ptmap is None):
        hdu5 = fits.ImageHDU(data=ptmap,header=modelhdr.copy())
        #hdu5.header = modelhdr.copy()
        hdu5.name = 'PtSrcMap'
        hdu5.header.append(("Title","Pt Src Models"))
        hdu5.header.append(("Target",cluster))          
        hdu5.header.append(("XTENSION","What Mate"))
        hdu5.header.append(("SIMPLE","T")) 
        hdu5.header.append(("Units",units,'Map Units'))          
        hdu5.verify('fix')

        cmodmap = ptmap + modelmap
        
        hdu6 = fits.ImageHDU(data=cmodmap,header=modelhdr.copy())
        #hdu6.header = modelhdr.copy()
        hdu6.name = 'AllModelMap'
        hdu6.header.append(("Title","All Models"))
        hdu6.header.append(("Target",cluster))          
        hdu6.header.append(("XTENSION","What Mate"))
        hdu6.header.append(("SIMPLE","T")) 
        hdu6.header.append(("Units",units,'Map Units'))          
        hdu6.verify('fix')
        myhdu.extend([hdu5,hdu6])
        

    if not(datamap is None):
        hdu1 = fits.ImageHDU(data=datamap,header=modelhdr.copy())
        #hdu1.header = modelhdr.copy()
        hdu1.name = 'DataMap'
        hdu1.header.append(("Title","Data Map"))
        hdu1.header.append(("Target",cluster))          
        hdu1.header.append(("XTENSION","What Mate"))
        hdu1.header.append(("SIMPLE","T")) 
        hdu1.header.append(("MNLVL",mnlvl,'Assumed/fitted mean level')) 
        hdu1.header.append(("Units",units,'Map Units'))          
        hdu1.verify('fix')

        residual = datamap - modelmap - ptmap - mnlvl
        hdu2 = fits.ImageHDU(data=residual,header=modelhdr.copy())
        #hdu2.header = modelhdr.copy()
        hdu2.name = 'Residual'
        hdu2.header.append(("Title","Residual"))
        hdu2.header.append(("Target",cluster))          
        hdu2.header.append(("XTENSION","What Mate"))
        hdu2.header.append(("SIMPLE","T")) 
        hdu2.header.append(("Units",units,'Map Units'))          
        hdu2.verify('fix')
        
        myhdu.extend([hdu1,hdu2])
        
        if not(rmsmap is None):
            pix_sigma = smooth/(pixsize*sig2fwhm)
            smres = scipy.ndimage.filters.gaussian_filter(residual, pix_sigma)
            #smres = residual*1.0 # It's already smoothed...
            nzi = (rmsmap > 0)
            rsnr      = smres*0.0
            rsnr[nzi] = smres[nzi] / rmsmap[nzi]
            hdu3 = fits.ImageHDU(data=rsnr,header=modelhdr.copy())
            #hdu3.header = modelhdr.copy()
            hdu3.name = 'ResSNRMap'
            hdu3.header.append(("Title","Residual SNR Map"))
            hdu3.header.append(("Target",cluster))          
            hdu3.header.append(("XTENSION","What Mate"))
            hdu3.header.append(("SIMPLE","T")) 
            hdu3.header.append(("Units",'SNR','Map Units'))          
            hdu3.verify('fix')
            myhdu.append(hdu3)
            
    if not(rmsmap is None):
        hdu4 = fits.ImageHDU(data=rmsmap,header=modelhdr.copy())
        #hdu4.header = modelhdr.copy()
        hdu4.name = 'RMSMap'
        hdu4.header.append(("Title","RMS Map"))
        hdu4.header.append(("Target",cluster))          
        hdu4.header.append(("XTENSION","What Mate"))
        hdu4.header.append(("SIMPLE","T")) 
        hdu4.header.append(("Units",units,'Map Units'))          
        hdu4.verify('fix')
        myhdu.append(hdu4)

    if not(nmap is None):
        hdu7 = fits.ImageHDU(data=nmap,header=modelhdr.copy())
        #hdu7.header = modelhdr.copy()
        hdu7.name = 'NoiseMap'
        hdu7.header.append(("Title","Noise Map"))
        hdu7.header.append(("Target",cluster))          
        hdu7.header.append(("XTENSION","What Mate"))
        hdu7.header.append(("SIMPLE","T")) 
        hdu7.header.append(("Units",units,'Map Units'))          
        hdu7.verify('fix')
        myhdu.append(hdu7)

    if not(fullpath is None):
        #import pdb;pdb.set_trace()
        hdulist = fits.HDUList(myhdu)
        hdulist.writeto(fullpath,overwrite=True)

    return myhdu

def make_fits_from_fits(cluster,pos=None,ppbins=None,edges=None,version='-SVD10_Mar13_Trim6-',
                        dataset='M2',inst='MUSTANG2',mytype='NP',session=1,domnlvl=True,ptamps=None,
                        iteration=1,m2fwhm=10.7,rfwhm=0.0,geom=[0,0,0,1,1,1,0,0]):
    
    outdir = '/home/romero/Results_Python/Rings/'+cluster+'/Iter/'
    version = mytype+version  #+'Iter'+str(session)
    prename = dataset+'_'+cluster+'_'+version+'_'
    inmap,inhdr,rmsmap,rmshdr, xymap, bins,map_vars, noisemap = get_map_products(cluster='Zw3146',iteration=iteration)
    mapshape = inmap.shape
    if pos is None:
        solns = np.load(outdir+prename+'Solutions.npy') # Or something like this.
        pos   = solns[:,0]
    if ppbins is None:
        ppbins = bins

    #print xymap.shape
    #import pdb;pdb.set_trace()
        
    modelmap = make_model_map(pos,ppbins,edges,map_vars,mapshape,inst=inst,mytype=mytype,mycluster=cluster,slopes=None,
                              xymap=xymap,geom=geom)

    if not (ptamps is None):
        ptmap = make_ptsrc_map(cluster,ptamps,xymap,inhdr,rfwhm=rfwhm)
        ptmap  = ptmap.reshape(inmap.shape)
        #modelmap += ptmap
    
    mnlvl=pos[-1] if domnlvl else 0.0
    print mnlvl
    
    modelmap  = modelmap.reshape(inmap.shape)
    ### Already done...
    #pix_fwhm  = m2fwhm/2.0     # For Zwicky 3146, the FWHM was 10.7
    #sig2fwhm  = np.sqrt(8.0*np.log(2.0))
    #pix_sigma = pix_fwhm/sig2fwhm
    #smoothedmodel = scipy.ndimage.filters.gaussian_filter(modelmap, pix_sigma) + mnlvl

    fullpath = outdir+prename+'Residual.fits'
    myhdu = make_hdu(modelmap,inhdr,ptmap=ptmap,datamap=inmap,rmsmap=rmsmap,nmap=noisemap,cluster=cluster,mnlvl=mnlvl,
                     smooth=10.0,fullpath=fullpath,geom=geom)


def make_ptsrc_map(cluster,ptamps,xymap,header,gpfile=None,pixs=2.0,rfwhm=0.0):

    if gpfile is None:
        gpfile = gpfile_by_clus(cluster)
        
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
    for j,myfwhm in enumerate(fwhm):
        if rfwhm > myfwhm:
            sigma[j] = rfwhm/sig2fwhm
        #sigma = np.ones(len(sigma))*rfwhm/sig2fwhm
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

def just_print_soln(dataset='M2',version='-Mar14_v2-',cluster='Zw3146',model='NP',domnlvl=True,doemcee=False,session=0,
                    nslice=8,slicenum=0,ring_combine=False,slope=False,slices=False,fpause=False,makemap=True,iteration=0,ySph=False):

    version = model+version
    ### You might want "_Iter" to be "_Session" depending on what you are actually doing...
    outdir = '/home/romero/Results_Python/Rings/'+cluster+'/'
    if slicenum > 0:
        outdir = outdir+'Slices/'
    if session > 0:
        version = version+'Session'+str(session)+'-'
        outdir = outdir+'BySession/'
    if iteration > 0:
        outdir = outdir+'Iter/'
        version = version+'Iter'+str(iteration)
        
    if slicenum > 0:
        #session=slicenum
        version = version+'Slice'+str(slicenum)
        slices=True
    #if nslice > 0:
    #    version = version+'Slice'+str(slicenum+1)
    #    slices=True
    ### dataset can be either 'M2' or 'M2_ACT'

    mymnlvl=3.0e-5 if domnlvl else 0.0

    slopes=None
    if ring_combine:
        gdata,gcurv,gedge      = combine_rings(version=version,cluster=cluster,nslice=nslice)
    else:
        curve,deriv,edges,trim,nsrc,geom = load_rings(option=dataset,cluster=cluster,session=session,iteration=iteration)
        #gdata,gcurv,gedge      = get_data_cov_edges(curve,deriv,edges,trim=0,slices=slices,nslice=nslice,slicenum=slicenum,
        #                                            nsrc=0,fpause=fpause)
        gdata,gcurv,gedge      = get_data_cov_edges(curve,deriv,edges,trim=10,slices=slices,nslice=nslice,slicenum=slicenum,
                                                    nsrc=6,fpause=False,ptsrcamps=True)
        print(gdata)
    
def map_minus_rings(dataset='M2',version='-Apr_SphRings_5c-',cluster='Zw3146',model='NP',domnlvl=False,doemcee=False,session=0,
                    nslice=8,slicenum=0,ring_combine=False,slope=False,slices=False,fpause=False,makemap=True,iteration=1,
                    ptamps=None,fwhm=10.7,pixsize=2.0,rfwhm=0.0):

    version = model+version
    ### You might want "_Iter" to be "_Session" depending on what you are actually doing...
    outdir = '/home/romero/Results_Python/Rings/'+cluster+'/'
    if slicenum > 0:
        outdir = outdir+'Slices/'
    if session > 0:
        version = version+'Session'+str(session)+'-'
        outdir = outdir+'BySession/'
    if iteration > 0:
        outdir = outdir+'Iter/'
        version = version+'Iter'+str(iteration)
        
    if slicenum > 0:
        #session=slicenum
        version = version+'Slice'+str(slicenum)
        slices=True
    #if nslice > 0:
    #    version = version+'Slice'+str(slicenum+1)
    #    slices=True
    ### dataset can be either 'M2' or 'M2_ACT'

    inmap,inhdr,rmsmap,rmshdr, xymap, bins,map_vars, noisemap = get_map_products(cluster='Zw3146',iteration=iteration)
    sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)

    mymnlvl=1.0e-7 if domnlvl else 0.0

    slopes=None
    if ring_combine:
        gdata,gcurv,gedge      = combine_rings(version=version,cluster=cluster,nslice=nslice)
    else:
        curve,deriv,edges,trim,nsrc,geom = load_rings(option=dataset,cluster=cluster,session=session,iteration=iteration)
        gdata,gcurv,gedge      = get_data_cov_edges(curve,deriv,edges,trim=trim,slices=slices,nslice=nslice,slicenum=slicenum,
                                                    nsrc=nsrc,fpause=fpause)
        if slope:
            #gdata,gcurv,gedge      = get_better_slope_rings(option=dataset,cluster=cluster,session=session)
            slopes = get_slope(option=dataset,cluster=cluster,mypass=iteration,slices=slices,gedge=gedge,trim=trim,
                               nsrc=nsrc,fpause=fpause,session=session)
            print 'Your slopes are: ',slopes
            print '=============================================================='

    nrings    = len(gedge)-1
    if slopes is None:
        slopes = np.zeros((nrings))
    edgy_rads = gedge * np.pi / (180.0*3600.0)
    yProf   = np.zeros(map_vars['thetas'].shape)
    for i in range(nrings):
        gRads1 = (map_vars['thetas'] >=  edgy_rads[i]) # Good radii (condition 1)
        gRads2 = (map_vars['thetas'] < edgy_rads[i+1]) # Good radii (condition 2)
        gRads  = [gRad1 and gRad2 for gRad1,gRad2 in zip(gRads1,gRads2)]
        myrads = map_vars['thetas'][gRads]
        a      = 1.0 - slopes[i]*(myrads-edgy_rads[i]) * (3600.0*180.0/np.pi)
        print(i,slopes[i]*3600*180/np.pi,np.min(a),np.max(a),np.max(myrads-edgy_rads[i]))
        #import pdb;pdb.set_trace()
        yProf[gRads] = gdata[i]*a
    skymap    = grid_profile(map_vars["thetas"], yProf, xymap, geoparams=geom)
    skymap    = skymap.reshape(inmap.shape)
    sig2fwhm  = np.sqrt(8.0*np.log(2.0)) 
    pix_sigma = fwhm/(pixsize*sig2fwhm)
    #pix_sigma = fwhm/(sig2fwhm)
    #mymap     = scipy.ndimage.filters.gaussian_filter(skymap, pix_sigma)
    #import pdb;pdb.set_trace()
    mymap     = imf.fourier_filtering_2d(skymap,'gauss',fwhm/pixsize)
    modelmap  = mymap.flatten()
    #modelmap  = skymap.flatten()
    
    ptamps = get_data_cov_edges(curve,deriv,edges,trim=trim,slices=slices,nslice=nslice,slicenum=slicenum,
                                nsrc=nsrc,fpause=fpause,ptsrcamps=True)
    print(ptamps)
    #import pdb;pdb.set_trace()
    
    if not (ptamps is None):
        ptmap = make_ptsrc_map(cluster,ptamps,xymap,inhdr,rfwhm=rfwhm)
        ptmap  = ptmap.reshape(inmap.shape)
        #import pdb;pdb.set_trace()
        
    #mnlvl=mymnlvl if domnlvl else 0.0
    #print mnlvl
    
    modelmap  = modelmap.reshape(inmap.shape)
    #pix_fwhm  = m2fwhm/2.0     # For Zwicky 3146, the FWHM was 10.7
    #sig2fwhm  = np.sqrt(8.0*np.log(2.0))
    #pix_sigma = pix_fwhm/sig2fwhm
    #smoothedmodel = scipy.ndimage.filters.gaussian_filter(modelmap, pix_sigma) + mnlvl
    
    prename = dataset+'_'+cluster+'_'+version+'_'

    fullpath = outdir+prename+'DataMinusRings.fits'
    myhdu = make_hdu(modelmap,inhdr,ptmap=ptmap,datamap=inmap,rmsmap=rmsmap,cluster=cluster,mnlvl=mymnlvl,
                     smooth=10.0,fullpath=fullpath,geom=geom)

    if fpause:
        import pdb; pdb.set_trace()
        
def gpfile_by_clus(cluster):

    if cluster == 'Zw3146':
        gpdir='/home/data/MUSTANG2/AGBT17_Products/Zw3146/Minkasi/GaussPars/'
        gpfile = gpdir+'zwicky_6src_gaussp_3Feb2019_minchi_all_v2.npy'
        #gpfile = gpdir+'zwicky_8src_gaussp_25Mar2019_minchi_all_v1.npy'
    if cluster == 'MACS0717':
        gpdir='/home/data/MUSTANG2/AGBT17_Products/MACS0717/Minkasi/GaussPars/'
        gpfile=gpdir+'M0717_3src_gaussp_2019-04-01_minchi_all_v1.npy'
    if cluster == 'MACS1149':
        gpdir='/home/data/MUSTANG2/AGBT17_Products/MACS1149/Minkasi/GaussPars/'
        gpfile=gpdir+'MACS1149_1src_gaussp_2019-04-01_minchi_all_v0.npy'

    return gpfile
