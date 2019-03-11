import numpy as np
from scipy import linalg as sla

class Sampler:

    def __init__(self,ndim,funclnp,fac=0.1,nchains=1,myseed=496,noseed=False):
        if not noseed: np.random.seed(myseed)
        self.ndim       = ndim
        self.k          = nchains
        self.iterations = 0
        self._lnprob    = np.empty((nchains,0))
        self.lnpfunc    = funclnp
        self.chain      = np.empty((nchains,0,ndim))
        #self._xchain    = np.empty((nchains,1))
        self.blobs      = np.empty((0,nchains,1))
        self._rotmat    = np.diag(np.ones(ndim))*fac
        self.params     = np.zeros(ndim)
        self.nbd        = 1
        self.acc_chain  = np.zeros((nchains,1))
        
    def sample(self,cparams,iterations=1,pcov=None):

        clnp,cxtras   = self._get_lnp(cparams)
        self.params   = cparams
        self.chain    = np.asarray(cparams).reshape(self.k,1,self.ndim)
        #import pdb;pdb.set_trace()
        nbd           = len(cxtras[0])
        self.nbd      = nbd
        self.blobs    = np.asarray(cxtras).reshape(1,self.k,self.nbd)
        self._lnprob  = np.asarray([clnp]).reshape(self.k,1)
        self._rotmat  = get_cho(pcov) if type(pcov) != type(None) else np.matmul(self._rotmat,cparams)

        #self._chain= np.concatenate((self.chain,np.zeros((self.k, 1, self.ndim))),axis=1)

        for j in range(int(iterations-1)):
            self.iterations += 1        
            params     = self.make_step(cparams)
            lnp, xtras = self._get_lnp(params)
            accept     = np.exp(lnp - clnp) > np.random.uniform()
            #import pdb;pdb.set_trace()
            for i,acc in enumerate(accept):
                if acc:
                    cparams[i]=params[i];cxtras[i]=xtras[i];clnp[i]=lnp[i]
            self.chain  = np.concatenate((self.chain,np.asarray(cparams).reshape(self.k,1,self.ndim)),axis=1)
            self._lnprob = np.concatenate((self._lnprob,np.asarray([clnp]).reshape(self.k,1)),axis=1)
            #import pdb;pdb.set_trace()
            self.blobs = np.concatenate((self.blobs,np.asarray(cxtras).reshape(1,self.k,self.nbd)),axis=0)
            #import pdb;pdb.set_trace()
            self.acc_chain  = np.concatenate((self.acc_chain,np.asarray(accept).reshape(self.k,1)),axis=0)

            yield cparams, clnp, cxtras
        
    def make_step(self,p):

        retp=[]
        for myp in p:
            #unistep = np.random.uniform(0,1,self.ndim)
            norstep = np.random.normal(0,1,self.ndim)
            my_step = np.matmul(self._rotmat,norstep)
            #print my_step
            #import pdb;pdb.set_trace()
            retp.append(myp + my_step)
            
        return retp

    def _get_lnp(self,pos=None):

        p = self.params if pos is None else pos
        results = list(map(self.lnpfunc, [p[i] for i in range(len(p))]))
        try:
            lnprob = np.array([float(l[0]) for l in results])
            blob = [l[1] for l in results]
        except (IndexError, TypeError):
            lnprob = np.array([float(l) for l in results])
            blob = None

        return lnprob, blob

def get_cov(pcovfile=None):
    
    #deffile= '/home/romero/Results_Python/Rings/M2_Zw3146_NP_v1_Parameter_Covariance.npy'
    defdir = '/home/data/MUSTANG2/AGBT17_Products/Zw3146/Minkasi'
    deffile= defdir+'/M2_Zw3146_NP_v1_Parameter_Covariance.npy'
    myfile = deffile if type(pcovfile) == type(None) else pcovfile
    mycov  = np.load(myfile)
    
    return mycov

def cfac(pcovfile=None):

    return get_cho(get_cov(pcovfile))
    
def get_cho(mycov):
    mycho  = sla.cholesky(mycov,lower=True)
    rotmat = np.transpose(mycho)

    return rotmat
