import numpy as np                      # A useful package...
import astropy.units as u             # U just got imported!
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import get_data_info as gdi           # Not much of a joke to make here.
gdi=reload(gdi)
import os, corner, emcee_stats
import analytic_integrations as ai
import scipy.stats as stats
import PowerLawBin as PLB
import numerical_integration as ni    #


gfontsize=20  # Global font size

def plot_steps(sampler,outdir,filename,burn_in=200):

    nwalk,nstep,ndim = sampler.chain.shape
    #burn_in          = 200
    
    stepmap    = plt.figure(3,figsize=(20,ndim),dpi=200); plt.clf()
    myfontsize = 10
    pos_comps  = set(['bulk','mnlvl'])
    
    for i in range(ndim):
        ax = stepmap.add_subplot(ndim+1,1,i+1)
        #import pdb;pdb.set_trace()
        ax.plot(np.array([sampler.chain[:,j,i] for j in range(nstep)]))
        isgtz = (sampler.chain[:,:,i] > 0)
        isgtz1d = isgtz.reshape(np.product(isgtz.shape))
        ax.get_xaxis().set_visible(False) # Maybe works?
        #import pdb;pdb.set_trace()
        if i == 0:
            ylims = ax.get_ylim()
            yval  = (ylims[1] - ylims[0])*0.5 + ylims[1]
            #for j,comp in enumerate(pos_comps):
            #    xval = float(nstep*j)/(len(pos_comps)+0.5)
            #    ax.text(xval, yval, comp, color=assoc_colo[comp],fontsize=20)
            xval = float(nstep*len(pos_comps))/(len(pos_comps)+0.5)
            ax.text(xval, yval, "likelihood", color="b",fontsize=20)

        
        if all(isgtz1d):
            ax.set_yscale("log", nonposy='clip')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
            #ax.get_yaxis().set_visible(False) # Maybe works?
        ax.set_ylabel((r'$P_{0}$',r'$P_{1}$',r'$P_{2}$',r'$P_{3}$',r'$P_{4}$',r'$P_{5}$',r'$P_{6}$',r'$P_{7}$',
                       r'$P_{8}$',r'$P_{9}$',r'$P_{10}$',r'$P_{11}$',r'$P_{12}$',r'$P_{13}$',r'$P_{14}$',
                       r'$P_{15}$',r'$P_{16}$',r'$P_{17}$',r'$P_{18}$',r'$P_{19}$',r'$P_{20}$',r'$P_{21}$',
                       r'$P_{22}$',r'$P_{23}$',r'$P_{24}$')[i],fontsize=myfontsize)
    ax = stepmap.add_subplot(ndim+1,1,ndim+1)
    ### This is just sampler._lnprob.T, that is the transpose. WHY...NOT?
    #ax.plot(np.array([sampler._lnprob[:,j] for j in range(nsteps)]),"b")
    ax.plot(sampler._lnprob.T,"b")
    myyr = [np.min(sampler._lnprob[:,burn_in*2:])*1.1,np.max(sampler._lnprob[:,burn_in*2:])*0.9]
    ax.set_ylim(myyr)
    ax.set_ylabel(r'$ln(\mathcal{L}$)',fontsize=myfontsize)
  
    plt.xlabel('Steps',fontsize=myfontsize)
    #filename = "step.png"
    fullpath = os.path.join(outdir,filename)
#2106_MUSTANG_6_B_Real_200S_40B_ML-NO_PP-NO_POWER_20W/
#    fullpath='/home/romero/Results_Python/plots_to_show/MUSTANG_Real_step.png'
    plt.savefig(fullpath)
    #plt.close(stepmap)
    plt.close()

def plot_pres_bins(solns,dataset,outdir,filename,cluster='Zw3146',runits='arcseconds',punits=r'keV / cm$^{-3}$',
                   IntegratedYs=None,overlay='None',mymodel='NP',bare=False,solns2=None,rads2=None,notes2=None,
                   ySph=False,geom=[0,0,0,1,1,1,0,0]):

    sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)
    fwhm1,norm1,fwhm2,norm2,fwhm,smfw,freq,FoV = gdi.inst_params('MUSTANG2')
    rin = fwhm.to("arcsec").value / 2.0; rout = FoV.to("arcsec").value / 2.0; axcol = 'r'
    ################################################################################
    
    myfontsize=6
    r500_ang = (map_vars['r500'].decompose() / map_vars['d_ang']).decompose()
    rinasec  = (r500_ang * u.rad.to('arcsec') ).value
    p500     = map_vars['p500']
    
    myfig = plt.figure(1,figsize=(5,3),dpi=300)
    plt.clf()
    ax = myfig.add_subplot(111)
    ax.axvline(rinasec,color='k', linestyle ="dotted")    # "original" R500

    radii =  np.logspace(np.log10(rin / 10.0), np.log10(rout*10.0),200)
    if type(IntegratedYs) != type(None):
        #print IntegratedYs.shape
        #import pdb;pdb.set_trace()

        rfromy,rinasec,m5_pm,thisR500,rlas,rhas,r500,m500,p500 = get_pressure_plot_addons(IntegratedYs,map_vars,ySph=ySph)
        #print p500
        #print map_vars['p500'].to('keV cm**-3')
        #print r500
        #print map_vars['r500'].to('kpc')

        #import pdb;pdb.set_trace()
                                                                                          
    if mymodel == 'NP':
    
        psolns   = solns[:-1,:]/Pdl2y       # Should now be in units of keV cm^-3
        errs     = [psolns[:,2].value,psolns[:,1].value]
        pressure = psolns[:,0].value
        arc_rads = bins*3600.0*180.0/np.pi    # In arcseconds
        mylabel = "Deprojected profile" if solns2 is None else "From Minkasi"
        ax.errorbar(arc_rads,pressure,yerr=errs,fmt='.',label=mylabel,capsize=5,color='royalblue')
        pprof,alphas = ai.log_profile(pressure,list(arc_rads),radii)
        #rdiff = np.abs(radii - rinasec)
        #rind  = (rdiff == np.min(rdiff))
        #r5alp = alphas[rind]
        #import pdb; pdb.set_trace()
        ax.set_xlim((np.min(arc_rads)/3.0,np.max(arc_rads)*3.0))
        ax.set_ylim((np.min(pressure)/5.0,np.max(pressure)*5.0))
        #ax.plot(radii, pprof, label='Power law interpolation')
        ax.plot(radii, pprof,color='royalblue')

        if not (solns2 is None):
            nbins     = len(rads2)
            psolns2   = solns2[1:nbins+1,:]/Pdl2y       # Should now be in units of keV cm^-3
            errs2     = [psolns2[:,2].value,psolns2[:,1].value]
            pressure2 = psolns2[:,0].value
            ax.errorbar(rads2,pressure2,yerr=errs2,fmt='.',label="From IDL",capsize=5,color='green')
            pprof2,alphas2 = ai.log_profile(pressure2,list(rads2),radii)
            ax.plot(radii, pprof2,color='green')


    if mymodel == 'GNFW':

        vars1 = solns[:-1,:]
        vars2 = vars1[:,0]
        myup     = np.array([1.177,8.403,5.4905,0.3081,1.0510])
        #myvars = vars2[:-1]
        if len(vars2) < len(myup):
            #myup[1:1+len(vars2)]=vars2
            myup[0:len(vars2)]=vars2
        else:
            myup = vars2    # If len(pos) > 5, that's OK...we won't use those!

        R500  = map_vars['r500'].to('kpc')
        P500  = map_vars['p500'].to('keV cm**-3')
        print R500,P500
        #R500  = r500.to('kpc')
        #P500  = p500.to('keV cm**-3')
        #print R500,P500
        #r500  = (R500 / map_vars['d_ang']).decompose().value
        physr = (radii* map_vars['d_ang']).to('kpc') * np.pi / (180.0 * 3600.0)
        #physr = (map_vars['thetas']* map_vars['d_ang']).to('kpc')
        pprof = gdi.gnfw(R500, P500, physr, c500=myup[0], p=myup[1], a=myup[4], b=myup[2], c=myup[3])
        print Pdl2y
        #ulp   = (pprof).decompose().value
        ulp   = (pprof).to('keV cm**(-3)').value
        ax.plot(radii, ulp, color='b',label='Fitted gNFW')

    if mymodel == 'Beta':

        vars1 = solns[:-1,:]
        vars2 = vars1[:,0]
        #miscvar=1.0
        #import pdb;pdb.set_trace()
        
        R500  = map_vars['r500'].to('kpc')
        P500  = map_vars['p500'].to('keV cm**-3')
        #R500  = r500.to('kpc')
        #P500  = p500.to('keV cm**-3')
        r500  = (R500 / map_vars['d_ang']).decompose().value
        physr = (radii* map_vars['d_ang']).to('kpc') * np.pi / (180.0 * 3600.0)
        #physr = (map_vars['thetas']* map_vars['d_ang']).to('kpc')
        pprof = vars2[0]*(1.0+(physr.value/vars2[1])**2)**(-1.5*vars2[2])        ### Beta model
        #pprof = gdi.gnfw(R500, P500, physr, c500=myup[0], p=myup[1], a=myup[4], b=myup[2], c=myup[3])
        #import pdb;pdb.set_trace()
        ulp   = pprof * Pdl2y.to('cm**3 / keV').value
        ax.plot(radii, ulp, color='b',label='Fitted Beta Model')
       
        
    ax.axvline(rin,color=axcol, linestyle ="dashed")      # HWHM
    ax.axvline(rout,color=axcol, linestyle ="dashed")     # Half the FOV
    ax.set_yscale("log")                                  #
    ax.set_xscale("log")                                  #
    
    ### Make things visible!
    ax.set_xlabel("Radius ("+runits+")",fontsize=myfontsize)
    ax.set_ylabel("Pressure ("+punits+")",fontsize=myfontsize)
    #axarr[plotind].tick_params('both',labelsize=5)
    ax.tick_params('both',labelsize=5)
    #ax.set_title(mycluster.name+r'($ M_{500} = $'+m5final+')',fontsize=myfontsize)
    ax.grid()

    if type(IntegratedYs) != type(None):
        #rfromy,rinasec,m5_pm,thisR500,rlas,rhas,r500,m500,p500 = get_pressure_plot_addons(IntegratedYs,map_vars)
        print 'ThisR500 is ',thisR500
        ax.axvline(thisR500,color='k', linestyle ="dotted")  # A gray scale for original R500
        if bare == False:
            ax.axvline(rinasec,color='g', linestyle ="dotted")  # A gray scale for original R500
            ax.axvline(rlas,color='0.75', linestyle ="dotted")   # A gray scale for original R500
            ax.axvline(rhas,color='0.75', linestyle ="dotted")   # A gray scale for original R500
            ax.set_title(cluster+r' ($ M_{500} = $'+m5_pm+')',fontsize=myfontsize*1.1,y=1.07)
        else:
            ax.set_title(cluster,fontsize=myfontsize*1.1,y=1.07)
    else:
        ax.set_title(cluster,fontsize=myfontsize*1.1,y=1.07)

    ax2 = ax.twiny()
    ax2.set_xlim(tuple([xlim/rinasec for xlim in ax.get_xlim()]))
    ax2.set_xscale("log")
    ax2.tick_params('x',labelsize=5)
    ax3 = ax.twinx()
    p5scale = p500.to('keV cm**-3').value
    ax3.set_ylim(tuple([ylim/p5scale for ylim in ax.get_ylim()]))
    ax3.set_ylabel(r"Pressure ($P_{500}$)",fontsize=myfontsize)
    ax3.set_yscale("log")
    ax3.tick_params('y',labelsize=5)
    #ax.set_title(mycluster.name+r' ($ M_{500} = $'+m5_pm+')',fontsize=myfontsize*1.1,y=1.07)

    if bare: overlay='a10'

    #overlay='None'
    if overlay == 'a10':
        #rads = map_vars['d_ang']*bins   # bins should already be in radians
        opm500 = (rfromy,p500,m500)
        my_a10 = overplot_a10(map_vars,noleg=False,myfs=5,my500=opm500,myax=ax,bare=bare,geom=geom)
        #import pdb;pdb.set_trace()
        
    if overlay == 'XMM':
        overplot_XMM(myax=ax)
    
    #filename = tstr+"pressure_"+compname+".png"
    #fullpath = os.path.join(hk.hk_outs.newpath,hk.hk_outs.prefilename+filename)
    fullpath = os.path.join(outdir,filename)
    plt.savefig(fullpath)
    #filename = tstr+"pressure_"+compname+".eps"
    #fullpath = os.path.join(hk.hk_outs.newpath,hk.hk_outs.prefilename+filename)
    #plt.savefig(fullpath,format='eps')
    #import pdb;pdb.set_trace()
    print(fullpath)
    plt.close()

def plot_surface_profs(model,bins,gdata,gcurve,gedge,outdir,filename,cluster='Zw3146',
                       runits='arcseconds',overlay='None',mymodel='NP',pinit=None,
                       bare=False,slopes=None,geom=[0,0,0,1,1,1,0,0]):

    sz_vars, map_vars, def_bins, Pdl2y = gdi.get_underlying_vars(cluster)
    pbins=bins
    if mymodel == 'Beta' or mymodel == 'GNFW': pbins = map_vars['thetas']*180.0*3600.0/np.pi
    myfontsize=6    
    myfig = plt.figure(1,figsize=(7,5),dpi=300)
    plt.clf()
    ax = myfig.add_subplot(111)
    mult = 1.0e6  # Multiply almost everything by this to be in microK
    
    cov   = np.linalg.inv(gcurve)
    myrms = np.sqrt(np.diag(cov))* mult


    if slopes is None:
        ldata = [[gdd,gdd] for gdd in gdata]
    else:
        #ldata = [[gdd,gdd-myslope*(ed2-ed1)] for ed1,ed2,gdd,myslope in
        #         zip(gedge[:-1],gedge[1:],gdata,slopes)]
        ldata = [[gdd,gdd*(1.0-myslope*(ed2-ed1))] for ed1,ed2,gdd,myslope in
                 zip(gedge[:-1],gedge[1:],gdata,slopes)]
    ledge = [[ed1,ed2] for ed1,ed2 in zip(gedge[:-1],gedge[1:])]
    lmids = [(ed1+ed2)/2.0 for ed1,ed2 in zip(gedge[:-1],gedge[1:])]
    lmode = [[bmo,bmo] for bmo in model]
    pedge = np.asarray(ledge); pedge=pedge.flatten() 
    pdata = np.asarray(ldata); pdata=pdata.flatten()* mult
    pmode = np.asarray(lmode); pmode=pmode.flatten()* mult
    pmids = np.asarray(lmids); pmids=pmids.flatten()
    
    #import pdb;pdb.set_trace()
    ax.plot(pedge, pdata, color='b',label='Input Data')
    if slopes is None:
        ax.errorbar(pmids,pdata[::2],yerr=myrms,fmt='.',color='c',capsize=5)
    else:
        ax.errorbar(pedge[::2],pdata[::2],yerr=myrms,fmt='.',color='c',capsize=5) 
    if bare == False: ax.plot(pedge, pmode, color='r',label='Best Fit')
    ax.set_ylabel(r"Brightness ($\mu$K)",fontsize=myfontsize*2)
    ax.set_xlabel("Radius ("+runits+")",fontsize=myfontsize*2)

    if type(pinit) != type(None) and bare == False:
        linit = [[pi,pi] for pi in pinit]
        ppin = np.asarray(linit); ppin=ppin.flatten()* mult
        ax.plot(pedge,ppin,color='k',label='Initial Estimate')
    
    fullpath = os.path.join(outdir,filename)
    plt.legend()
    plt.savefig(fullpath)

    
def plot_correlations(samples,outdir,filename,blobs=None,cluster='Zw3146',mtype='NP',domnlvl=True):

    myfontsize=6
    plt.figure(2,figsize=(20,12))
    plt.clf()
    sz_vars, map_vars, bins, Pdl2y = gdi.get_underlying_vars(cluster)

    pars2plot = samples*1.0
    if domnlvl == False:
        pars2plot = pars2plot[:,:-1]
    deflabels=[r'$P_{1}$',r'$P_{2}$',r'$P_{3}$',r'$P_{4}$',r'$P_{5}$',r'$P_{6}$',r'$P_{7}$',r'$P_{8}$',
               r'$P_{9}$',r'$P_{10}$',r'$P_{11}$',r'$P_{12}$',r'$P_{13}$',r'$P_{14}$',r'$P_{15}$',r'$P_{16}$',
               r'$P_{17}$',r'$P_{18}$',r'$P_{19}$',r'$P_{20}$',r'$P_{21}$',r'$P_{22}$',r'$P_{23}$',r'$P_{24}$']
    p2pshape = pars2plot.shape
    print p2pshape
    npars    = p2pshape[1]
    mylabels = deflabels[:npars]

    mycolors = ['k' for i in range(len(mylabels))]
    myfracs = [0.999 for i in range(len(mylabels))]
    mycount=1
    
    for i in range(p2pshape[1]):
        mymed     = np.median(pars2plot[:,i])
        medabsdev = np.median(np.abs(pars2plot[:,i] - mymed))/0.6745
        myvind    = 1
        myvar     = 'P'
        ss = '{'+'{0:d},{1:d}'.format(myvind,i-mycount+1)+'}'
        vprec = r'${p}_{ss}$'.format(p=myvar,ss=ss)
        if i < npars-1:
            if mtype == 'GNFW':
                myvar = [r'$C_{500}$',r'$P_0$',r'$\beta$',r'$\gamma$',r'$\alpha$'][i]
                vprec = myvar   #ss='1'
            if mtype == 'Beta':
                myvar = [r'$P_0$',r'$r_s$',r'$\beta$'][i]
                vprec = myvar   #ss='1'
        else:
            myvar='Mn'
            ss = 'lvl'
            vprec = r'${p}_{ss}$'.format(p=myvar,ss=ss)

        if mymed < 1e-1 or mymed > 1e2:
            order = np.floor(np.log10(np.abs(mymed)))
            pars2plot[:,i] /= (10.0**order)
            #mylabels[i] = "$P_{%s}/10^{%s}$" % (str(i),str(order))
            #mylabels[i] = r'{p}$_{ss}/10^{{{rs}}}$'.format(p=myvar,ss=ss,rs=int(order))
            ocorr = r'$10^{{{rs}}}$'.format(rs=int(order))
            mylabels[i] = vprec+'/'+ocorr
        else:
            #mylabels[i] = r'{p}$_{ss}$'.format(p=myvar,ss=ss)
            mylabels[i] = vprec


    
    if type(blobs) != type(None):
        gblind = (blobs > 0) ; bblind = (blobs <= 0)
        ### Initial values of Y_int are with radians**2, rather than Mpc**2
        be5    = blobs * (map_vars['hofz'].value**(-1./3) * map_vars['d_ang'].to('Mpc').value)**2
        
        #be5[gblind] = np.log10(blobs[gblind])+5.0
        be5[gblind] = np.log10(blobs[gblind])
        #print np.min(be5),np.max(be5)
        blobiness = blobs.shape
        #foo = [r'$\log_{10}(Y_{cyl,%s})+5}$' % str(count+1) for count in range(blobiness[1])]
        #foo = [r'$\log(Y_{%s})+5}$' % str(count+1) for count in range(blobiness[1])]
        foo = []; mybounds=[]
        for i in range(blobiness[1]):
            mymed = np.median(be5[:,i])
            myexp = np.ceil(mymed)
            be5[:,i] -= myexp;  mymed -= myexp
            foo.append(r'$\log(Y_{%s})+{%s}$' % (str(i),str(int(-myexp))))

            #if mymed > 15:
            #    be5[:,i] -= 15
            #    mymed -= 15
            be5pos = (be5[:,i] > mymed); be5neg = (be5[:,i] < mymed)
            medabspos = np.median(np.abs(be5[be5pos,i] - mymed))/0.6745
            medabsneg = np.median(np.abs(be5[be5neg,i] - mymed))/0.6745
            mymin = np.max([mymed - 4.0*medabsneg,np.min(be5[:,i])-medabsneg])
            mymax = np.min([mymed + 4.0*medabspos,np.max(be5[:,i])+medabspos])
            print medabspos,medabsneg, mymed, mymin,mymax
            mybounds.append( (mymin, mymax) )
            mycolors.append('b')

        pars2plot = np.hstack((pars2plot,be5))

        mylabels.extend(foo)
        print '#################################################################'
        print mylabels
        myfracs.extend([0.999 for i in range(len(foo))])

    print myfracs
    #h1dkeys={'color':'b'}
    #lblkeys={'color':mycolors}
    #print h1dkeys
    fig = corner.corner(pars2plot, bins = 45,quantiles=[0.16,0.50,0.84],labels=mylabels,
                        fontsize=myfontsize,show_titles=True,title_fmt=".2f",range=myfracs)
    fullpath = os.path.join(outdir, filename)

    plt.savefig(fullpath)
    plt.close()

def get_pressure_plot_addons(IntegratedYs,map_vars,ySph=False):

    initR500 = map_vars['r500'].decompose() / map_vars['d_ang']
    thisR500 = initR500.decompose().value * 3600*180 / np.pi  # Now in arcseconds
    
    Yarr     = IntegratedYs[:,0]
    Yints    = np.percentile(Yarr, [16, 50, 84],axis=0)
    yinteg   = Yints[1]       # This is the median...i.e. the best-fit value

    rlow ,m500_l,p500_l,msys_l = gdi.rMP500_from_y500(Yints[0],map_vars,ySZ=True,ySph=ySph) # ySZ basically denotes whether the angular distance
    rmed ,m500_m,p500_m,msys_m = gdi.rMP500_from_y500(Yints[1],map_vars,ySZ=True,ySph=ySph) # is already multiplied in or not.
    rhigh,m500_h,p500_h,msys_h = gdi.rMP500_from_y500(Yints[2],map_vars,ySZ=True,ySph=ySph)

    fb   = 0.9**((3.0*1.78+1.0)/(3.0*1.78))  # calibration factor, and how it feeds into Y-M uncertainty
    
    rcal,m5ca_l,p5ca_l,msyc_l = gdi.rMP500_from_y500(Yints[1]*fb,map_vars,ySZ=True,ySph=ySph)
    rcal,m5ca_h,p5ca_h,msyc_h = gdi.rMP500_from_y500(Yints[1]/fb,map_vars,ySZ=True,ySph=ySph)

    rinasec   = rmed * u.rad.to('arcsec')
    r500Mpc   = rmed * map_vars['d_ang'].to('Mpc')

    rlas,rhas  = rlow * u.rad.to('arcsec'), rhigh * u.rad.to('arcsec')

    #import pdb;pdb.set_trace()
    
    m5errs = np.array([m500_m.value-m500_l.value,m500_h.value-m500_m.value])
    m5cale = np.array([m500_m.value-m5ca_l.value,m5ca_h.value-m500_m.value])
    m5_pm   = pos_neg_formatter(m500_m.value,m5errs[1],m5errs[0],sys=msys_m.value,cal=m5cale)
    #m5_pm   = pos_neg_formatter(m500_m.value,m5errs[1],m5errs[0])

    return rmed,rinasec,m5_pm,thisR500,rlas,rhas,r500Mpc,m500_m,p500_m

def pos_neg_formatter(med,high_err,low_err,sys=None,cal=None):
    """
    Input the median (or mode), and the *error bars* (not percentile values, but the
    distance between the +/-1 sigma percentiles and the 0 sigma percentile).

    """

    mypow = np.floor(np.log10(med))
    myexp = 10.0**mypow

    if mypow > 0:
        psign = '+'
        pStr = psign+str(int(mypow))
    else:
        pStr = str(int(mypow))
        
    msig  = med/myexp
    hsig  = high_err/myexp
    lsig  = low_err/myexp

    
    msStr = "{:.2F}".format(msig)
    hsStr = '+'+"{:.2F}".format(hsig)
    lsStr = "{:.2F}".format(-lsig)

    baStr = r'${0}^{{{1}}}_{{{2}}}$'.format(msStr,hsStr,lsStr)

    if not (sys is None):
        hyStr = '+'+"{:.2F}".format(sys/myexp)
        lyStr = "{:.2F}".format(-sys/myexp)
        baStr = baStr + r' $^{{{0}}}_{{{1}}}$'.format(hyStr,lyStr)
    if not (sys is None):
        hyStr = '+'+"{:.2F}".format(cal[1]/myexp)
        lyStr = "{:.2F}".format(-cal[0]/myexp)
        baStr = baStr + r' $^{{{0}}}_{{{1}}}$'.format(hyStr,lyStr)
    
    exStr = 'E'+pStr
    coStr = baStr+exStr

    return coStr

def overplot_XMM(myax=None,noleg=False,myfs=5):

    XMM_file = '/home/data/X-ray/XMM/Zwicky3146_pressure.dat'
    XMM_cols = np.loadtxt(XMM_file, comments='#')
    plotpres = XMM_cols[:,2]
    ploterrs = XMM_cols[:,3]

    runif    = ((XMM_cols[:,1]**3 + XMM_cols[:,0]**3)/2.0)**(1.0/3.0)
    alphas   = ( (np.log(plotpres[1:])-np.log(plotpres[:-1])) /
                 (np.log(runif[1:])-np.log(runif[:-1])) )
    mypoly   = np.polyfit((runif[1:]+runif[:-1])/2.0,alphas,1)
    galphf   = np.poly1d(mypoly)
    galphin  = galphf(XMM_cols[:,0])
    galphout = galphf(XMM_cols[:,1])

    galphs   = (galphin + galphout)/2.0
    #foo      = (XMM_cols[:,0]**(3+galphin))
    #bar      = foo**(1.0/galphs)
    mypow    = (2+galphs)
    rwpl     = ( (XMM_cols[:,0]**mypow + XMM_cols[:,1]**mypow)/2.0)**(1.0/mypow)
    
    print rwpl
    print '--------------------------------------------------------------------------------'
    print galphs
    print '--------------------------------------------------------------------------------'
    print galphout
    print '--------------------------------------------------------------------------------'
    print galphin
    
    #plt.plot((runif[1:]+runif[:-1])/2.0,alphas)
    #plt.plot(runif,galphout)
    #plt.show()
    
    #plotrads = rwted * 60.0 # Now in arcseconds
    plotrads = rwpl  * 60.0 # Now in arcseconds

    #ax.errorbar(arc_rads,pressure,yerr=errs,fmt='.',label="Deprojected profile",capsize=5)

    if type(myax) != type(None):
        myax.errorbar(plotrads,plotpres,yerr=ploterrs,color="purple",label="XMM",capsize=5) #,fontsize=myfontsize
        #myax.set_xlim((np.min(plotrads)/3.0,np.max(plotrads)*3.0))
    else:
        plt.errorbar(plotrads,plotpres,yerr=ploterrs,color="purple",label="XMM",capsize=5) #,fontsize=myfontsize
        #plt.xlim((np.min(plotrads)/3.0,np.max(plotrads)*3.0))
    if noleg == False:
        myax.legend(fontsize=myfs)

    
def overplot_a10(map_vars,noleg=False,myfs=5,my500=(None,None,None),myax=None,bare=False,geom=[0,0,0,1,1,1,0,0]): 

    radnx   = np.array([1.0,300.0]) * (u.arcsec).to('rad')
    #if type(ax) != type(None):
    #    radnx = [xxx for xxx in ax.get_xlim()]
    nbins   = 100
    #oprads  = np.logspace(np.log10(radnx[0]),np.log10(radnx[1]), nbins)
    oprads  = map_vars['thetas']
    physrads= oprads * map_vars['d_ang']
    plotrads= oprads * (u.rad).to('arcsec')

    a10pres   = gdi.a10_from_m500_z(map_vars["m500"], map_vars["z"], physrads)
    plotpres  = a10pres.to('keV cm**-3').value

    #print np.max(plotpres),np.min(plotpres),physrads[0]

    if not bare:
        if type(myax) != type(None):
            myax.plot(plotrads,plotpres,"tab:purple",label="Initial guess A10") #,fontsize=myfontsize
            myax.set_xlim((np.min(plotrads)/3.0,np.max(plotrads)*3.0))
        else:
            plt.plot(plotrads,plotpres,"tab:purple",label="Initial guess A10") #,fontsize=myfontsize
            plt.xlim((np.min(plotrads)/3.0,np.max(plotrads)*3.0))

    if type(my500[0]) != type(None):
        my_rad, p500, m500 = my500     # my_rad is *my* R500 in radians
        unit_a10 = gdi.a10_from_m500_z(m500, map_vars["z"], physrads)
        my_a10   = unit_a10.to('keV cm**-3').value
        szcv,szcu = gdi.get_sz_values()

        Pdl2y     = (szcu['thom_cross']*map_vars['d_ang']/szcu['m_e_c2']).to("cm**3 keV**-1")
        unitless_profile = (unit_a10*Pdl2y).decompose().value
        
        yProf = ni.int_profile(oprads, unitless_profile, oprads)
        yint ,newr500=PLB.Y_SZ_via_scaling(yProf,oprads,map_vars['r500'],map_vars['d_ang'],geom) # As of Aug. 31, 2018
        #print newr500,(map_vars['r500']/map_vars['d_ang']).decompose()
        #import pdb;pdb.set_trace()

        #my_a10 = cpp.a10_gnfw(p500,my_rad*u.rad/u.rad,hk.av.mycosmo,oprads)
        if type(myax) != type(None):
            myax.plot(plotrads,my_a10,"g",label="A10 from Y500") #,fontsize=myfontsize
        else:
            plt.plot(plotrads,my_a10,"g",label="A10 from Y500") #,fontsize=myfontsize

    #import pdb;pdb.set_trace()
            
    if noleg == False:
        myax.legend(fontsize=myfs)
        #plt.legend(fontsize=myfs)

    return my_a10
    
def plot_ConvTests(ConvTests,outdir,mpp=8,name='Cluster'):

    """
    mpp is the    *M*aximum *P*er *P*lot ... i.e. number of parameters plotted per plot.

    """
    myfontsize = gfontsize/3
    myshape    = ConvTests['GoodmanWeare2010'].shape 
    nsubplots  = int(np.ceil(myshape[1]/float(mpp)))
    print nsubplots
    if nsubplots < 1: import pdb;pdb.set_trace()
    xlen       = 6+nsubplots
    ylen       = nsubplots*2
    fig, axarr = plt.subplots(nsubplots,figsize=(5,3),dpi=300, sharex=True)
    inclnlike=True
    mylabels   = get_par_labels(cluster=name,inclnlike=inclnlike)
    print 'Length of mylabels is ', len(mylabels)
    
    if nsubplots == 1: axarr = [axarr]
    altlim = myshape[1] if inclnlike == True else myshape[1]-1
    
    for i in range(nsubplots):
        
        uplim    = np.min([(i+1)*mpp,altlim]) 
        #myyvals1 = ConvTests['GoodmanWeare2010'][:,i*mpp:uplim]
        #myyvals2 = ConvTests['Fardal_emcee'][:,i*mpp:uplim]
        #mylabels1= mylabels[i*mpp:uplim]
        print 'Upper limit is ',uplim

        for j in range(uplim-i*mpp):
            my_y1 = ConvTests['GoodmanWeare2010'][:,i*mpp+j]
            my_y2 = ConvTests['Fardal_emcee'][:,i*mpp+j]
            axarr[i].plot(ConvTests['Abscissa'],my_y1,label=mylabels[i*mpp+j])
            axarr[i].plot(ConvTests['Abscissa'],my_y2,linestyle='--')
            print mylabels[i*mpp+j],j
            
        if i == 0: axarr[i].set_title(name,fontsize=myfontsize)
        axarr[i].set_ylabel("$f$",fontsize=myfontsize)
        axarr[i].legend(fontsize=myfontsize)
        if i == nsubplots-1:
            axarr[i].set_xlabel("Steps",fontsize=myfontsize)

            
    filename = "Convergence_Tests.png"
    fullpath = os.path.join(outdir,filename)
    #print fullpath
    plt.savefig(fullpath)
    #plt.show()

def get_par_labels(cluster='Zw3146',blobs=None,inclnlike=False):

    compnames  = ['bulk','bulk','bulk','bulk','bulk','bulk','mnlvl']
    pos_comps  = set(compnames)
    mycompco   = [0,1,2,3,4,5,0]
    compabbr   = {'mnlvl':'K','bulk':'P','ptsrc':'S','blob':'A'}
    pos_colors = ['k','g','r','c','m','y']
    assoc_colo = {}   

    npars      = len(compnames)
    mylabels   = [' ' for i in range(npars)]
    #print 'wassup'
    
    for i in range(npars):
        myvar     = compabbr[compnames[i]]
        myvind    = mycompco[i]+1
        whcomp    = [compname == compnames[i] for compname in compnames]
        whccount  = [compcount == mycompco[i] for compcount in mycompco]
        whboth    = [cond1 and cond2 for cond1,cond2 in zip(whcomp,whccount)]
        mycount   = 0
        for j in range(len(whboth)):
            if whboth[j]:
                mycount=j
                break
            
        ss = '{'+'{0:d},{1:d}'.format(myvind,i-mycount+1)+'}'
        mylabels[i] = r'${p}_{ss}$'.format(p=myvar,ss=ss)

    #print 'pussaw'
    if type(blobs) != type(None):
        blobiness = blobs.shape
        foo = [r'$Y_{cyl,%s}}$' % str(count+1) for count in range(blobiness[1])]
        mylabels.extend(foo)

    if inclnlike == True:
        mylabels.extend([r'$\ln(\mathcal{L})$'])

    return mylabels

def plot_autocorrs(sampler,outdir,filename,burn_in=200):

    nwalk,nstep,ndim = sampler.chain.shape
    nsigma = stats.norm.isf(1.0/nstep)
    
    stepmap    = plt.figure(3,figsize=(20,ndim),dpi=200); plt.clf()
    myfontsize = 10
    pos_comps  = set(['bulk','mnlvl'])
    
    for i in range(ndim):
        ax = stepmap.add_subplot(ndim+1,1,i+1)
        mystd = 0
        for j in range(nwalk):
            autocorr = emcee_stats.autocorr_func_1d(sampler.chain[j,:,i])
            stddev   = np.std(autocorr)
            if stddev > mystd: mystd = stddev 
            ax.plot(autocorr)

            
        ax.set_ylim([-nsigma*mystd,nsigma*mystd])
        ax.get_xaxis().set_visible(False) # Maybe works?
        #import pdb;pdb.set_trace()
        if i == 0:
            ylims = ax.get_ylim()
            yval  = (ylims[1] - ylims[0])*0.5 + ylims[1]
            #for j,comp in enumerate(pos_comps):
            #    xval = float(nstep*j)/(len(pos_comps)+0.5)
            #    ax.text(xval, yval, comp, color=assoc_colo[comp],fontsize=20)
            xval = float(nstep*len(pos_comps))/(len(pos_comps)+0.5)
            ax.text(xval, yval, "likelihood", color="b",fontsize=20)

        #if all(isgtz1d):
        #    ax.set_yscale("log", nonposy='clip')
        #    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
            #ax.get_yaxis().set_visible(False) # Maybe works?
        ax.set_ylabel((r'$P_{0}$',r'$P_{1}$',r'$P_{2}$',r'$P_{3}$',r'$P_{4}$',r'$P_{5}$',r'$P_{6}$',r'$P_{7}$',
                       r'$P_{8}$',r'$P_{9}$',r'$P_{10}$',r'$P_{11}$',r'$P_{12}$',r'$P_{13}$',r'$P_{14}$',
                       r'$P_{15}$',r'$P_{16}$',r'$P_{17}$',r'$P_{18}$',r'$P_{19}$',r'$P_{20}$',r'$P_{21}$',
                       r'$P_{22}$',r'$P_{23}$',r'$P_{24}$')[i],fontsize=myfontsize)
    ax = stepmap.add_subplot(ndim+1,1,ndim+1)
    ### This is just sampler._lnprob.T, that is the transpose. WHY...NOT?
    #ax.plot(np.array([sampler._lnprob[:,j] for j in range(nsteps)]),"b")
    probcorr = emcee_stats.autocorr_func_1d(sampler._lnprob.flatten())
    ax.plot(probcorr,"b")
    #myyr = [np.min(sampler._lnprob[:,burn_in*2:])*1.1,np.max(sampler._lnprob[:,burn_in*2:])*0.9]
    #ax.set_ylim(myyr)
    ax.set_ylabel(r'$ln(\mathcal{L}$)',fontsize=myfontsize)
  
    plt.xlabel('Autocorrelations',fontsize=myfontsize)
    #filename = "step.png"
    fullpath = os.path.join(outdir,filename)
#2106_MUSTANG_6_B_Real_200S_40B_ML-NO_PP-NO_POWER_20W/
#    fullpath='/home/romero/Results_Python/plots_to_show/MUSTANG_Real_step.png'
    plt.savefig(fullpath)
    #plt.close(stepmap)
    plt.close()

def plot_surface_profs_v2(idlvals,idlrads,gdata,gcurve,gedge,outdir,filename,cluster='Zw3146',
                          runits='arcseconds',overlay='None',mymodel='NP',pinit=None,
                          bare=False,slopes=None):

    sz_vars, map_vars, def_bins, Pdl2y = gdi.get_underlying_vars(cluster)
    myfontsize=6    
    myfig = plt.figure(1,figsize=(7,5),dpi=300)
    plt.clf()
    ax = myfig.add_subplot(111)
    mult = 1.0e6  # Multiply almost everything by this to be in microK
    
    cov   = np.linalg.inv(gcurve)
    myrms = np.sqrt(np.diag(cov))* mult


    if slopes is None:
        ldata = [[gdd,gdd] for gdd in gdata]
    else:
        ldata = [[gdd,gdd-myslope*(ed2-ed1)] for ed1,ed2,gdd,myslope in
                 zip(gedge[:-1],gedge[1:],gdata,slopes)]
    ledge = [[ed1,ed2] for ed1,ed2 in zip(gedge[:-1],gedge[1:])]
    lmids = [(ed1+ed2)/2.0 for ed1,ed2 in zip(gedge[:-1],gedge[1:])]
    pedge = np.asarray(ledge); pedge=pedge.flatten() 
    pdata = np.asarray(ldata); pdata=pdata.flatten()* mult
    pmids = np.asarray(lmids); pmids=pmids.flatten()
    
    #import pdb;pdb.set_trace()
    ax.plot(pedge, pdata, color='b',label='Minkasi Rings')
    if slopes is None:
        ax.errorbar(pmids,pdata[::2],yerr=myrms,fmt='.',color='c',capsize=5)
    else:
        ax.errorbar(pedge[::2],pdata[::2],yerr=myrms,fmt='.',color='c',capsize=5)
        
    if bare == False: ax.plot(idlrads, idlvals*mult, color='r',label='IDL SB profile')
    ax.set_ylabel(r"Brightness ($\mu$K)",fontsize=myfontsize*2)
    ax.set_xlabel("Radius ("+runits+")",fontsize=myfontsize*2)

    if type(pinit) != type(None) and bare == False:
        linit = [[pi,pi] for pi in pinit]
        ppin = np.asarray(linit); ppin=ppin.flatten()* mult
        ax.plot(pedge,ppin,color='k',label='Initial Estimate')
    
    fullpath = os.path.join(outdir,filename)
    plt.legend()
    plt.savefig(fullpath)
