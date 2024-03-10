###50% all Arctan
#+jump
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, rfft2, irfft2, cartdist,enkf_update,gaspcohn, bulk_ensrf

from EnSF_Sparse_obs import EnSF
import torch

# horizontal covariance localization length scale in meters.
hcovlocal_scale = 5800*1000 #float(sys.argv[1])

covinflate1 = 0.8
covinflate2 = -1
exptname = os.getenv('exptname','Sparse_EnSF_linear_64')
threads = int(os.getenv('OMP_NUM_THREADS','1'))

diff_efold = None # use diffusion from climo file

profile = False # turn on profiling?

use_letkf = True  # use LETKF
global_enkf = False # global EnSRF solve
read_restart = False
# if savedata not None, netcdf filename will be defined by env var 'exptname'
# if savedata = 'restart', only last time is saved (so expt can be restarted)
#savedata = True 
#savedata = 'restart'
savedata = None
#nassim = 101 
#nassim_spinup = 1
nassim = 300 # assimilation times to run
nassim_spinup = 100

direct_insertion = False 
if direct_insertion: print('# direct insertion!')

nanals = 20 # ensemble members

oberrstdev = 1. # ob error standard deviation in K
oberrstdev_arctan = 0.01

# nature run created using sqg_run.py.
filename_climo = 'sqg_N64_12hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = 'sqg_N64_12hrly.nc' # file name for nature run to draw obs
#filename_truth = 'sqg_N256_N96_12hrly.nc' # file name for nature run to draw obs

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
rsics = np.random.RandomState() # varying seed for initial conditions
rsarctan = np.random.RandomState(98) # fixed seed for observations
rsjump = np.random.RandomState(10) # fixed seed for observations

# get model info
nc_climo = Dataset(filename_climo)
# parameter used to scale PV to temperature units.
scalefact = nc_climo.f*nc_climo.theta0/nc_climo.g
# initialize qg model instances for each ensemble member.
x = nc_climo.variables['x'][:]
y = nc_climo.variables['y'][:]
x, y = np.meshgrid(x, y)
nx = len(x); ny = len(y)
dt = nc_climo.dt
if diff_efold == None: diff_efold=nc_climo.diff_efold
pvens = np.empty((nanals,2,ny,nx),np.float32)
if not read_restart:
    pv_climo = nc_climo.variables['pv']
    indxran = rsics.choice(pv_climo.shape[0],size=nanals,replace=False)
else:
    ncinit = Dataset('%s_restart.nc' % exptname, mode='r', format='NETCDF4_CLASSIC')
    ncinit.set_auto_mask(False)
    pvens[:] = ncinit.variables['pv_b'][-1,...]/scalefact
    tstart = ncinit.variables['t'][-1]
    #for nanal in range(nanals):
    #    print(nanal, pvens[nanal].min(), pvens[nanal].max())
# get OMP_NUM_THREADS (threads to use) from environment.
models = []
for nanal in range(nanals):
    if not read_restart:
        pvens[nanal] = pv_climo[indxran[nanal]]
        #print(nanal, pvens[nanal].min(), pvens[nanal].max())
    pvens[nanal] = pv_climo[0] + np.random.normal(0,1000,size=(2,ny,nx))
    models.append(\
    SQG(pvens[nanal],
    nsq=nc_climo.nsq,f=nc_climo.f,dt=dt,U=nc_climo.U,H=nc_climo.H,\
    r=nc_climo.r,tdiab=nc_climo.tdiab,symmetric=nc_climo.symmetric,\
    diff_order=nc_climo.diff_order,diff_efold=diff_efold,threads=threads))
if read_restart: ncinit.close()

# vertical localization scale
Lr = np.sqrt(models[0].nsq)*models[0].H/models[0].f
vcovlocal_fact = gaspcohn(np.array(Lr/hcovlocal_scale))
#vcovlocal_fact = 0.0 # no increment at opposite boundary
#vcovlocal_fact = 1.0 # no vertical localization

print('# use_letkf=%s global_enkf=%s' % (use_letkf,global_enkf))
print("# hcovlocal=%g vcovlocal=%s diff_efold=%s covinf1=%s covinf2=%s nanals=%s" %\
     (hcovlocal_scale/1000.,vcovlocal_fact,diff_efold,covinflate1,covinflate2,nanals))

# if nobs > 0, each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
# if nobs < 0, fixed network of every Nth grid point used (N = -nobs)
nobs = 2048  #nx*ny//16 # number of obs to assimilate (randomly distributed)
#nobs = -1 # fixed network, every -nobs grid points. nobs=-1 obs at all pts.

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
# set up arrays for obs and localization function
if nobs < 0:
    nskip = -nobs
    if (nx*ny)%nobs != 0:
        raise ValueError('nx*ny must be divisible by nobs')
    nobs = (nx*ny)//nskip**2
    print('# fixed network nobs = %s' % nobs)
    fixed = True
else:
    fixed = False
    print('# random network nobs = %s' % nobs)
if nobs == nx*ny//2: fixed=True # used fixed network for obs every other grid point
print('fixed is', fixed)

oberrvar = oberrstdev**2*np.ones(nobs,float)
pvob = np.empty((2,nobs),float)
covlocal = np.empty((ny,nx),float)
covlocal_tmp = np.empty((nobs,nx*ny),float)
xens = np.empty((nanals,2,nx*ny),float)
if not use_letkf:
    obcovlocal = np.empty((nobs,nobs),float)
else:
    obcovlocal = None

if global_enkf: # model-space localization matrix
    n = 0
    covlocal_modelspace = np.empty((nx*ny,nx*ny),float)
    x1 = x.reshape(nx*ny); y1 = y.reshape(nx*ny)
    for n in range(nx*ny):
        dist = cartdist(x1[n],y1[n],x1,y1,nc_climo.L,nc_climo.L)
        covlocal_modelspace[n,:] = gaspcohn(dist/hcovlocal_scale)

obtimes = nc_truth.variables['t'][:]
if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    print('# restarting from %s.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))
print('# assim interval = %s secs (%s time steps)' % (assim_interval,assim_timesteps))
print('# ntime,pverr_a,pvsprd_a,pverr_b,pvsprd_b,obinc_b,osprd_b,obinc_a,obsprd_a,omaomb/oberr,obbias_b,inflation,tr(P^a)/tr(P^b)')

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[ntstart]
    models[nanal].timesteps = assim_timesteps

# initialize output file.
if savedata is not None:
   nc = Dataset('%s.nc' % exptname, mode='w', format='NETCDF4_CLASSIC')
   nc.r = models[0].r
   nc.f = models[0].f
   nc.U = models[0].U
   nc.L = models[0].L
   nc.H = models[0].H
   nc.nanals = nanals
   nc.hcovlocal_scale = hcovlocal_scale
   nc.vcovlocal_fact = vcovlocal_fact
   nc.oberrstdev = oberrstdev
   nc.g = nc_climo.g; nc.theta0 = nc_climo.theta0
   nc.nsq = models[0].nsq
   nc.tdiab = models[0].tdiab
   nc.dt = models[0].dt
   nc.diff_efold = models[0].diff_efold
   nc.diff_order = models[0].diff_order
   nc.filename_climo = filename_climo
   nc.filename_truth = filename_truth
   nc.symmetric = models[0].symmetric
   xdim = nc.createDimension('x',models[0].N)
   ydim = nc.createDimension('y',models[0].N)
   z = nc.createDimension('z',2)
   t = nc.createDimension('t',None)
   obs = nc.createDimension('obs',nobs)
   ens = nc.createDimension('ens',nanals)
   pv_t =\
   nc.createVariable('pv_t',np.float32,('t','z','y','x'),zlib=True)
   pv_c =\
   nc.createVariable('pv_c',np.float32,('t','ens','z','y','x'),zlib=True)
   pv_b =\
   nc.createVariable('pv_b',np.float32,('t','ens','z','y','x'),zlib=True)
   pv_a =\
   nc.createVariable('pv_a',np.float32,('t','ens','z','y','x'),zlib=True)
   pv_a.units = 'K'
   pv_b.units = 'K'
   pv_c.units = 'K'
   inf = nc.createVariable('inflation',np.float32,('t','z','y','x'),zlib=True)
   pv_obs = nc.createVariable('obs',np.float32,('t','obs'))
   x_obs = nc.createVariable('x_obs',np.float32,('t','obs'))
   y_obs = nc.createVariable('y_obs',np.float32,('t','obs'))
   # eady pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
   xvar = nc.createVariable('x',np.float32,('x',))
   xvar.units = 'meters'
   yvar = nc.createVariable('y',np.float32,('y',))
   yvar.units = 'meters'
   zvar = nc.createVariable('z',np.float32,('z',))
   zvar.units = 'meters'
   tvar = nc.createVariable('t',np.float32,('t',))
   tvar.units = 'seconds'
   ensvar = nc.createVariable('ens',np.int32,('ens',))
   ensvar.units = 'dimensionless'
   xvar[:] = np.arange(0,models[0].L,models[0].L/models[0].N)
   yvar[:] = np.arange(0,models[0].L,models[0].L/models[0].N)
   zvar[0] = 0; zvar[1] = models[0].H
   ensvar[:] = np.arange(1,nanals+1)

# initialize kinetic energy error/spread spectra
kespec_errmean = None; kespec_sprdmean = None

ncount = 0
nanals2 = 4 # ensemble members used for kespec spread

init_std_x_state = (pvens.reshape(nanals,2*nx*ny)).std(axis=0)

#Jump
jumpnoise = rsjump.normal(0,900,size=(1,2,ny,nx)) #3000 as avg baseline
jumpnoise_reshape =jumpnoise.reshape(2,ny*nx)  #3000
jump = np.where(rsjump.uniform(0,1,nassim) > 0.9)[0]

for ntime in range(300): #nassim

    # check model clock
    if models[0].t != obtimes[ntime+ntstart]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t, obtimes[ntime+ntstart]))

    t1 = time.time()
    if not fixed:
        # randomly choose points from model grid
        if nobs == nx*ny:
            indxob = np.arange(nx*ny)
        else:
            indxob = np.sort(rsobs.choice(nx*ny,nobs,replace=False))
    else:
        mask = np.zeros((ny,nx),bool)
        # if every other grid point observed, shift every other time step
        # so every grid point is observed in 2 cycles.
        if nobs == nx*ny//2:
            if ntime%2:
                mask[0:ny,1:nx:2] = True
            else:
                mask[0:ny,0:nx:2] = True
        else:
            mask[0:ny:nskip,0:nx:nskip] = True
        indxob = np.flatnonzero(mask)
    
    pickarctan = 2048
    arctan_index =  np.sort(rsarctan.choice(nobs, pickarctan, replace=False)) #2048 1024 4096
    arctan_index_ensf = indxob[arctan_index]
    arctan_index_ensf = np.concatenate((arctan_index_ensf, arctan_index_ensf+nx*ny), axis=None) 
    for k in range(2):
        # surface temp obs
        if (ntime == jump).any():
            pvob[k] = scalefact*(pv_truth[ntime+ntstart,k,:,:].ravel()[indxob] + jumpnoise_reshape[k,indxob])
        else:
            pvob[k] = scalefact*pv_truth[ntime+ntstart,k,:,:].ravel()[indxob]
        pvob[k,arctan_index] = np.arctan(pvob[k,arctan_index]) + rsobs.normal(scale=oberrstdev_arctan,size=int(pickarctan)) # add ob errors int(nobs/2)
        pvob[k,~np.isin(np.arange(len(pvob[k])), arctan_index)] += rsobs.normal(scale=oberrstdev,size=int(nobs - pickarctan)) # add ob errors
    
    if ntime == 246 or ntime == ntime == 245:
        for k in range(2):
            pvob[k] = scalefact*pv_truth[ntime+ntstart,k,:,:].ravel()[indxob]
            pvob[k,arctan_index] = np.arctan(pvob[k,arctan_index]) + 0.5*rsobs.normal(scale=oberrstdev_arctan,size=int(pickarctan)) # add ob errors int(nobs/2)
            pvob[k,~np.isin(np.arange(len(pvob[k])), arctan_index)] += 0.5* rsobs.normal(scale=oberrstdev,size=int(nobs - pickarctan)) # add ob errors
    
    xob = x.ravel()[indxob]
    yob = y.ravel()[indxob]
    # compute covariance localization function for each ob
    if not fixed or ntime == 0:
        for nob in range(nobs):
            dist = cartdist(xob[nob],yob[nob],x,y,nc_climo.L,nc_climo.L)
            covlocal = gaspcohn(dist/hcovlocal_scale)
            covlocal_tmp[nob] = covlocal.ravel()
            dist = cartdist(xob[nob],yob[nob],xob,yob,nc_climo.L,nc_climo.L)
            if not use_letkf: obcovlocal[nob] = gaspcohn(dist/hcovlocal_scale)

    # first-guess spread (need later to compute inflation factor)
    fsprd = ((pvens - pvens.mean(axis=0))**2).sum(axis=0)/(nanals-1)

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens = np.empty((nanals,2,nobs),float)
    for nanal in range(nanals):
        for k in range(2):
            hxens[nanal,k,...] = np.arctan(scalefact*pvens[nanal,k,...].ravel()[indxob]) # surface pv obs
    hxensmean_b = hxens.mean(axis=0)
    obsprd = ((hxens-hxensmean_b)**2).sum(axis=0)/(nanals-1)
    # innov stats for background
    obfits = pvob - hxensmean_b
    obfits_b = (obfits**2).mean()
    obbias_b = obfits.mean()
    obsprd_b = obsprd.mean()
    pvensmean_b = pvens.mean(axis=0).copy()
    if (ntime == jump).any():
        pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime+ntstart]+jumpnoise))**2
    else:
        pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime+ntstart]))**2
    pvsprd_b = ((scalefact*(pvensmean_b-pvens))**2).sum(axis=0)/(nanals-1)

    if savedata is not None:
        if savedata == 'restart' and ntime != nassim-1:
            pass
        else:
            pv_t[ntime] = pv_truth[ntime+ntstart]
            pv_b[ntime,:,:,:] = scalefact*pvens
            #pv_obs[ntime] = pvob
            x_obs[ntime] = xob
            y_obs[ntime] = yob

    # EnKF update
    EnSF_Update = EnSF(n_dim = nx*ny*2, ensemble_size = nanals ,eps_alpha=0.05, device= 'cuda' ,\
                   obs_sigma = oberrstdev, euler_steps = 1000, scalefact = nc_climo.f*nc_climo.theta0/nc_climo.g, init_std_x_state = init_std_x_state,  ISarctan=True)
    # create 1d state vector.
    xens = pvens.reshape(nanals,2,nx*ny)
    # update state vector.
    if direct_insertion and nobs == nx*ny:
        for nanal in range(nanals):
            xens[nanal] =\
            pv_truth[ntime+ntstart].reshape(2,nx*ny) + \
            rsobs.normal(scale=oberrstdev,size=(2,nx*ny))/scalefact
        xens = xens - xens.mean(axis=0) + \
        pv_truth[ntime+ntstart].reshape(2,nx*ny) + \
        rsobs.normal(scale=oberrstdev,size=(2,nx*ny))/scalefact
    else:
        # hxens,pvob are in PV units, xens is not 
        if global_enkf and not use_letkf:
            xens = bulk_ensrf(xens,indxob,pvob,oberrvar,covlocal_modelspace,vcovlocal_fact,scalefact)
        else:
            indxob_ensf = np.concatenate((indxob, indxob+nx*ny), axis=None)  
            xens =\
            EnSF_Update.state_update_normalized(x_input = xens.reshape(nanals,2*nx*ny),state_target_input = pv_truth[ntime+ntstart].reshape(2*nx*ny),\
                                                 obs_input = pvob.reshape(2*nobs),sparse_idx=indxob_ensf, current = ntime, arcindex=arctan_index_ensf)
            xens = xens.cpu().numpy()
    # back to 3d state vector
    pvens = xens.reshape((nanals,2,ny,nx))
    t2 = time.time()
    #print('cpu time for EnKF update',t2-t1)

    if savedata is not None:
        if savedata == 'restart' and ntime != nassim-1:
            pass
        else:
            pv_c[ntime,:,:,:] = scalefact*pvens

    # forward operator on posterior ensemble.
    for nanal in range(nanals):
        for k in range(2):
            hxens[nanal,k,...] = np.arctan(scalefact*pvens[nanal,k,...].ravel()[indxob]) # surface pv obs

    # ob space diagnostics
    hxensmean_a = hxens.mean(axis=0)
    obsprd_a = (((hxens-hxensmean_a)**2).sum(axis=0)/(nanals-1)).mean()
    # expected value is HPaHT (obsprd_a).
    obinc_a = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_a)).mean()
    # expected value is HPbHT (obsprd_b).
    obinc_b = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_b)).mean()
    # expected value R (oberrvar).
    omaomb = ((pvob-hxensmean_a)*(pvob-hxensmean_b)).mean()

    # posterior multiplicative inflation.
    pvensmean_a = pvens.mean(axis=0)
    pvprime = pvens-pvensmean_a
    asprd = (pvprime**2).sum(axis=0)/(nanals-1)
    asprd_over_fsprd = asprd.mean()/fsprd.mean()
    if covinflate2 < 0:
        # relaxation to prior stdev (Whitaker & Hamill 2012)
        asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
        inflation_factor = 1.+covinflate1*(fsprd-asprd)/asprd
    else:
        # Hodyss et al 2016 inflation (covinflate1=covinflate2=1 works well in perfect
        # model, linear gaussian scenario)
        # inflation = asprd + (asprd/fsprd)**2((fsprd/nanals)+2*inc**2/(nanals-1))
        inc = pvensmean_a - pvensmean_b
        inflation_factor = covinflate1*asprd + \
        (asprd/fsprd)**2*((fsprd/nanals) + covinflate2*(2.*inc**2/(nanals-1)))
        inflation_factor = np.sqrt(inflation_factor/asprd)
    pvprime = pvprime*inflation_factor
    #pvens = pvprime + pvensmean_a

    # print out analysis error, spread and innov stats for background
    if (ntime == jump).any():
        pverr_a = (scalefact*(pvensmean_a-pv_truth[ntime+ntstart]-jumpnoise))**2
    else:
        pverr_a = (scalefact*(pvensmean_a-pv_truth[ntime+ntstart]))**2
    pvsprd_a = ((scalefact*(pvensmean_a-pvens))**2).sum(axis=0)/(nanals-1)
    print("%s %g %g %g %g %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
     np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
     obinc_b,obsprd_b,obinc_a,obsprd_a,omaomb/oberrvar.mean(),obbias_b,inflation_factor.mean(),asprd_over_fsprd))

    # save data.
    if savedata is not None:
        if savedata == 'restart' and ntime != nassim-1:
            pass
        else:
            pv_a[ntime,:,:,:] = scalefact*pvens
            tvar[ntime] = obtimes[ntime+ntstart]
            inf[ntime] = inflation_factor
            nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals):
        pvens[nanal] = models[nanal].advance(pvens[nanal])
    t2 = time.time()
    #print('cpu time for ens forecast',t2-t1)

    # compute spectra of error and spread
    if ntime >= nassim_spinup:
        pvfcstmean = pvens.mean(axis=0)
        pverrspec = scalefact*rfft2(pvfcstmean - pv_truth[ntime+ntstart])
        psispec = models[0].invert(pverrspec)
        psispec = psispec/(models[0].N*np.sqrt(2.))
        kespec = (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real
        if kespec_errmean is None:
            kespec_errmean =\
            (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real
        else:
            kespec_errmean = kespec_errmean + kespec
        for nanal in range(nanals2):
            pvsprdspec = scalefact*rfft2(pvens[nanal] - pvfcstmean)
            psispec = models[0].invert(pvsprdspec)
            psispec = psispec/(models[0].N*np.sqrt(2.))
            kespec = (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real
            if kespec_sprdmean is None:
                kespec_sprdmean =\
                (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real/nanals2
            else:
                kespec_sprdmean = kespec_sprdmean+kespec/nanals2
        ncount += 1

if savedata: nc.close()

