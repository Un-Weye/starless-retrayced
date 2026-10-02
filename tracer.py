#!/usr/bin/python
import numpy as np
import numpy as xp
from queue import Empty
import multiprocessing as mp
mp.set_start_method("fork")
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndim
import imageio.v2 as spm
import random,sys,time,os
import datetime

import multiprocessing as multi
import ctypes

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#importing scene
try:
    import ConfigParser as configparser
except ImportError: # we're on python 3
    import configparser

import blackbody as bb
import bloom

import gc
import curses


#enums
METH_LEAPFROG = 0
METH_RK4 = 1

logger.debug(f"xp = {xp.__name__}")

#rough option parsing
LOFI = False

DISABLE_DISPLAY = 0
DISABLE_SHUFFLING = 0

NTHREADS = 4

DRAWGRAPH = True

OVERRIDE_RES = False

SCENE_FNAME = 'scenes/default.scene'

CHUNKSIZE = 9000

for arg in sys.argv[1:]:
    if arg == '-d':
        LOFI = True
        continue
    if (arg == '--no-graph'):
        DRAWGRAPH = False
        continue
    if arg == '--no-display':
        DISABLE_DISPLAY = 1
        continue
    if arg == '--no-shuffle':
        DISABLE_SHUFFLING = 1
        continue

    if (arg == '-o') or (arg == '--no-bs'):
        DRAWGRAPH = False
        DISABLE_DISPLAY = True
        DISABLE_SHUFFLING = True
        continue

    if (arg[0:2] == '-c'):
        CHUNKSIZE = int(arg[2:])
        continue

    if arg[0:2] == "-j":
        NTHREADS = int(arg[2:])
        continue

    if arg[0:2] == "-r":
        RESOLUTION = [int(x) for x in arg[2:].split('x')]
        OVERRIDE_RES = True
        if (len(RESOLUTION) != 2):
            logger.error('''error: resolution "%s" unreadable''', arg[2:])
            logger.error("please format resolution correctly (e.g.: -r640x480)")
            exit()
        continue

    if arg[0] == '-':
        logger.error("unrecognized option: %s", arg)
        exit()

    SCENE_FNAME = arg



if not os.path.isfile(SCENE_FNAME):
    logger.error("scene file \"%s\" does not exist", SCENE_FNAME)
    sys.exit(1)


defaults = {
            "Distort":"1",
            "Fogdo":"1",
            "Blurdo":"1",
            "Fogmult":"0.02",
            "Diskinner":"1.5",
            "Diskouter":"4",
            "Resolution":"160,120",
            "Diskmultiplier":"100.",
            "Gain":"1",
            "Normalize":"-1",
            "Blurdo":"1",
            "Bloomcut":"2.0",
            "Airy_bloom":"1",
            "Airy_radius":"1.",
            "Iterations":"1000",
            "Stepsize":"0.02",
            "Cameraposition":"0.,1.,-10",
            "Fieldofview":1.5,
            "Lookat":"0.,0.,0.",
            "Horizongrid":"1",
            "Redshift":"1",
            "sRGBOut":"1",
            "Diskintensitydo":"1",
            "sRGBIn":"1",
            }

cfp = configparser.ConfigParser(defaults)
logger.debug("Reading scene %s...", SCENE_FNAME)
cfp.read(SCENE_FNAME)


FOGSKIP = 1


METHOD = METH_RK4

#enums to avoid per-iteration string comparisons

ST_NONE = 0
ST_TEXTURE = 1
ST_FINAL = 2

st_dict = {
        "none":ST_NONE,
        "texture":ST_TEXTURE,
        "final":ST_FINAL
    }

DT_NONE = 0
DT_TEXTURE = 1
DT_SOLID = 2
DT_GRID = 3
DT_BLACKBODY = 4
DT_HYBRID = 5

dt_dict = {
        "none":DT_NONE,
        "texture":DT_TEXTURE,
        "solid":DT_SOLID,
        "grid":DT_GRID,
        "blackbody":DT_BLACKBODY,
        "hybrid":DT_HYBRID
    }

#this section works, but only if the .scene file is good
#if there's anything wrong, it's a trainwreck
#must rewrite
try:
    if not OVERRIDE_RES:
        RESOLUTION = [int(x) for x in cfp.get('lofi','Resolution').split(',')]
    NITER = int(cfp.get('lofi','Iterations'))
    STEP = float(cfp.get('lofi','Stepsize'))
except (KeyError, configparser.NoSectionError):
    logger.debug("error reading scene file: insufficient data in lofi section")
    logger.debug("using defaults.")


if not LOFI:
    try:
        if not OVERRIDE_RES:
            RESOLUTION = [int(x) for x in cfp.get('hifi','Resolution').split(',')]
        NITER = int(cfp.get('hifi','Iterations'))
        STEP = float(cfp.get('hifi','Stepsize'))
    except (KeyError, configparser.NoSectionError):
        logger.debug("no data in hifi section. Using lofi/defaults.")

try:
    CAMERA_POS = [float(x) for x in cfp.get('geometry','Cameraposition').split(',')]
    TANFOV = float(cfp.get('geometry','Fieldofview'))
    LOOKAT = xp.array([float(x) for x in cfp.get('geometry','Lookat').split(',')])
    UPVEC = xp.array([float(x) for x in cfp.get('geometry','Upvector').split(',')])
    DISTORT = int(cfp.get('geometry','Distort'))
    DISKINNER = float(cfp.get('geometry','Diskinner'))
    DISKOUTER = float(cfp.get('geometry','Diskouter'))

    #options for 'blackbody' disktexture
    DISK_MULTIPLIER = float(cfp.get('materials','Diskmultiplier'))
    DISK_ALPHA_MULTIPLIER = float(cfp.get('materials','Diskalphamultiplier'))
    DISK_INTENSITY_DO = int(cfp.get('materials','Diskintensitydo'))
    REDSHIFT = float(cfp.get('materials','Redshift'))

    GAIN = float(cfp.get('materials','Gain'))
    NORMALIZE = float(cfp.get('materials','Normalize'))

    BLOOMCUT = float(cfp.get('materials','Bloomcut'))



except (KeyError, configparser.NoSectionError):
    logger.debug("error reading scene file: insufficient data in geometry section")
    logger.debug("using defaults.")


try:
    HORIZON_GRID = int(cfp.get('materials','Horizongrid'))
    DISK_TEXTURE = cfp.get('materials','Disktexture')
    SKY_TEXTURE = cfp.get('materials','Skytexture')
    SKYDISK_RATIO = float(cfp.get('materials','Skydiskratio'))
    FOGDO = int(cfp.get('materials','Fogdo'))
    BLURDO = int(cfp.get('materials','Blurdo'))
    AIRY_BLOOM = int(cfp.get('materials','Airy_bloom'))
    AIRY_RADIUS = float(cfp.get('materials','Airy_radius'))
    FOGMULT = float(cfp.get('materials','Fogmult'))

    #perform linear rgb->srgb conversion
    SRGBOUT = int(cfp.get('materials','sRGBOut'))
    SRGBIN = int(cfp.get('materials','sRGBIn'))
except (KeyError, configparser.NoSectionError):
    logger.debug("error reading scene file: insufficient data in materials section")
    logger.debug("using defaults.")


# converting mode strings to mode ints
try:
    DISK_TEXTURE_INT = dt_dict[DISK_TEXTURE]
except KeyError:
    logger.debug("Error: %s is not a valid accretion disc rendering mode", DISK_TEXTURE)
    sys.exit(1)

try:
    SKY_TEXTURE_INT = st_dict[SKY_TEXTURE]
except KeyError:
    logger.debug("Error: %s is not a valid sky rendering mode", SKY_TEXTURE)
    sys.exit(1)


logger.debug("%dx%d", RESOLUTION[0], RESOLUTION[1])

#just ensuring it's an xp.array() and not a tuple/list
CAMERA_POS = xp.array(CAMERA_POS)


#ensure the observer's 4-velocity is timelike
#since as of now the observer is schwarzschild stationary, we just need to check
#whether he's outside the horizon.
if xp.linalg.norm(CAMERA_POS) <= 1.:
    logger.debug("Error: the observer's 4-velocity is not timelike.")
    logger.debug("(try placing the observer outside the event horizon)")
    sys.exit(1)


DISKINNERSQR = DISKINNER*DISKINNER
DISKOUTERSQR = DISKOUTER*DISKOUTER

#ensuring existence of tests directory
if not os.path.exists("tests"):
    os.makedirs("tests")

#GRAPH
if DRAWGRAPH:
    logger.debug("Drawing schematic graph...")
    g_diskout       = plt.Circle((0,0),DISKOUTER, fc='0.75')
    g_diskin        = plt.Circle((0,0),DISKINNER, fc='white')

    g_photon        = plt.Circle((0,0),1.5,ec='y',fc='none')
    g_horizon       = plt.Circle((0,0),1,color='black')
    g_cameraball    = plt.Circle((CAMERA_POS[2],CAMERA_POS[0]),0.2,color='black')

    figure = plt.gcf()

    ax = plt.gca()

    ax.cla()

    gscale = 1.1*xp.linalg.norm(CAMERA_POS)
    ax.set_xlim((-gscale,gscale))
    ax.set_ylim((-gscale,gscale))
    ax.set_aspect('equal')

    l = 100


    ax.plot([CAMERA_POS[2],LOOKAT[2]] , [CAMERA_POS[0],LOOKAT[0]] , color='0.05', linestyle='-')


    figure.gca().add_artist(g_diskout)
    figure.gca().add_artist(g_diskin)
    figure.gca().add_artist(g_horizon)
    figure.gca().add_artist(g_photon)
    figure.gca().add_artist(g_cameraball)


    logger.debug("Saving diagram...")
    figure.savefig('tests/graph.png')

    ax.cla()

# these need to be here
# convert from linear rgb to srgb
def rgbtosrgb(arr):
    logger.debug("RGB -> sRGB...")
    #see https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    mask = arr > 0.0031308
    arr[mask] **= 1/2.4
    arr[mask] *= 1.055
    arr[mask] -= 0.055
    arr[~mask] *= 12.92


# convert from srgb to linear rgb
def srgbtorgb(arr):
    logger.debug("sRGB -> RGB...")

    mask = arr > 0.04045

    arr[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    arr[~mask] = arr[~mask] / 12.92


logger.debug("Loading textures...")

if SKY_TEXTURE == 'texture':
    # load using Pillow and convert to float array in range [0,1]
    im = Image.open('textures/bgedit.jpg').convert('RGB')
    texarr_sky = xp.array(im, dtype=float) / 255.0

    if SRGBIN:
        # must do this before resizing to get correct results
        srgbtorgb(texarr_sky)

    if not LOFI:
        logger.debug("(zooming sky texture...)")

        # resize using PIL
        im2 = Image.fromarray(
            (texarr_sky * 255).clip(0, 255).astype(np.uint8)
        )
        im2 = im2.resize(
            (int(im2.width * 2.0), int(im2.height * 2.0)),
            resample=Image.BICUBIC
        )
        texarr_sky = xp.array(im2, dtype=float) / 255.0

        print("SKY TEXTURE:", texarr_sky.shape)

texarr_disk = None
if DISK_TEXTURE in ('texture', 'hybrid'):
        im = Image.open('textures/adisk.jpg').convert('RGB')
        texarr_disk = xp.array(im, dtype=float) / 255.0
if DISK_TEXTURE == 'test':
   im = Image.open('textures/adisktest.jpg').convert('RGB')
   texarr_disk = xp.array(im, dtype=float) / 255.0
if texarr_disk is not None:

    if SRGBIN:
        srgbtorgb(texarr_disk)


#defining texture lookup
def lookup(texarr,uvarrin): #uvarrin is an array of uv coordinates
    uvarr = xp.clip(uvarrin,0.0,0.999)

    uvarr[:,0] *= float(texarr.shape[1])
    uvarr[:,1] *= float(texarr.shape[0])

    uvarr = uvarr.astype(int)

    return texarr[  uvarr[:,1], uvarr[:,0] ]



logger.debug("Computing rotation matrix...")

# this is just standard CGI vector algebra

FRONTVEC = (LOOKAT-CAMERA_POS)
FRONTVEC = FRONTVEC / xp.linalg.norm(FRONTVEC)

LEFTVEC = xp.cross(UPVEC,FRONTVEC)
LEFTVEC = LEFTVEC/xp.linalg.norm(LEFTVEC)

NUPVEC = xp.cross(FRONTVEC,LEFTVEC)

viewMatrix = xp.zeros((3,3))

viewMatrix[:,0] = LEFTVEC
viewMatrix[:,1] = NUPVEC
viewMatrix[:,2] = FRONTVEC


#array [0,1,2,...,numPixels]
pixelindices = xp.arange(0,RESOLUTION[0]*RESOLUTION[1],1)

#total number of pixels
numPixels = pixelindices.shape[0]

logger.debug("Generated %d pixel flattened array.", numPixels)

#useful constant arrays
ones = xp.ones((numPixels))
ones3 = xp.ones((numPixels,3))
UPFIELD = xp.outer(ones,xp.array([0.,1.,0.]))

#random sample of floats
ransample = np.random.random_sample((numPixels))

def vec3a(vec): #returns a constant 3-vector array (don't use for varying vectors)
    return xp.outer(ones,vec)

def vec3(x,y,z):
    return vec3a(xp.array([x,y,z]))

def norm(vec):
    # you might not believe it, but this is the fastest way of doing this
    # there's a stackexchange answer about this
    return xp.sqrt(xp.einsum('...i,...i',vec,vec))

def normalize(vec):
    #return vec/ (xp.outer(norm(vec),xp.array([1.,1.,1.])))
    return vec / (norm(vec)[:,np.newaxis])

# an efficient way of computing the sixth power of r
# much faster than pow!
# np has this optimization for power(a,2)
# but not for power(a,3)!

def sqrnorm(vec):
    return xp.einsum('...i,...i',vec,vec)

def sixth(v):
    tmp = sqrnorm(v)
    return tmp*tmp*tmp


def RK4f(y,h2):
    f = xp.zeros(y.shape)
    f[:,0:3] = y[:,3:6]
    f[:,3:6] = - 1.5 * h2 * y[:,0:3] / xp.power(sqrnorm(y[:,0:3]),2.5)[:,np.newaxis]
    return f


# this blends colours ca and cb by placing ca in front of cb
def blendcolors(cb,balpha,ca,aalpha):
            #* xp.outer(aalpha, xp.array([1.,1.,1.])) + \
    #return  ca + cb * xp.outer(balpha*(1.-aalpha),xp.array([1.,1.,1.]))
    return  ca + cb * (balpha*(1.-aalpha))[:,np.newaxis]


# this is for the final alpha channel after blending
def blendalpha(balpha,aalpha):
    return aalpha + balpha*(1.-aalpha)


def saveToImg(arr,fname):
    logger.debug(" - saving %s...", fname)
    #copy
    imgout = xp.array(arr)
    #clip
    imgout = xp.clip(imgout,0.0,1.0)
    #rgb->srgb
    if SRGBOUT:
        rgbtosrgb(imgout)
    #unflattening
    imgout = imgout.reshape((RESOLUTION[1],RESOLUTION[0],3))
    plt.imsave(fname,imgout)

# this is not just for bool, also for floats (as grayscale)
def saveToImgBool(arr,fname):
    saveToImg(xp.outer(arr,xp.array([1.,1.,1.])),fname)


#for shared arrays

def tonumpyarray(mp_arr):
    a = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
    a = a.reshape((numPixels,3))
    return a




#PARTITIONING

#partition viewport in contiguous chunks

#CHUNKSIZE = 9000
if not DISABLE_SHUFFLING:
    np.random.shuffle(pixelindices)
chunks = xp.array_split(pixelindices,int(numPixels/CHUNKSIZE) + 1)

NCHUNKS = len(chunks)

logger.debug("Split into %d chunks of %d pixels each", NCHUNKS, chunks[0].shape[0])

total_colour_buffer_preproc_shared = multi.Array(ctypes.c_float, numPixels * 3)
total_colour_buffer_preproc = tonumpyarray(total_colour_buffer_preproc_shared)

#open preview window

if not DISABLE_DISPLAY:
    logger.debug("Opening display...")

    plt.ion()
    plt.imshow(total_colour_buffer_preproc.reshape((RESOLUTION[1],RESOLUTION[0],3)))
    plt.draw()


#shuffle chunk list (does very good for equalizing load)

random.shuffle(chunks)


#partition chunk list in schedules for single threads

schedules = []

#from http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
q,r = divmod(NCHUNKS, NTHREADS)
indices = [q*i + min(i,r) for i in range(NTHREADS+1)]

for i in range(NTHREADS):
    schedules.append(chunks[ indices[i]:indices[i+1] ])



logger.debug("Split list into %d schedules with %s chunks each", NTHREADS, ", ".join([str(len(s)) for s in schedules]))




# global clock start
start_time = time.time()

itcounters = [0 for i in range(NTHREADS)]
chnkcounters = [0 for i in range(NTHREADS)]

#killers

killers = [False for i in range(NTHREADS)]


# command line output

class Outputter:
    def name(self, num):
        if num == -1:
            return "M"
        else:
            return str(num)

    def __init__(self):
        self.message = {}
        self.queue = multi.Queue()
        self.have_curses = False

        try:
            import curses
            self.curses = curses
            self.stdscr = curses.initscr()
            curses.noecho()
            self.have_curses = True
        except Exception:
            # headless or no curses available
            self.have_curses = False

        for i in range(NTHREADS):
            self.message[i] = "..."

        self.message[-1] = "..."
        self._display_started = False

    def start_display(self):
        if not self._display_started:
            print("\033[?1049h", end="", flush=True)
            print("\033[2J\033[H", end="", flush=True)
            self._display_started = True

    def stop_display(self):
        if self._display_started:
            print("\033[?25h", end="", flush=True)
            print("\033[?1049l", end="", flush=True)
            self._display_started = False


    def doprint(self):
        if self.have_curses:
            for i in range(NTHREADS + 1):
                line = self.name(i - 1) + "] " + self.message[i - 1]
                self.stdscr.move(i, 0)
                self.stdscr.clrtoeol()
                self.stdscr.addstr(i, 0, line)

            self.stdscr.refresh()

        else:
            self.start_display()

            lines = [
                f"{self.name(i)}: {self.message[i]}"
                for i in sorted(self.message)
            ]

            # Go to the top-left of the alternate screen.
            print("\033[H", end="")

            # Draw the complete status screen.
            for line in lines:
                print("\033[2K" + line)

            # Put the cursor back at the top for the next refresh.
            print(f"\033[{len(lines)}A", end="", flush=True)

    def parsemessages(self):
        doref = False

        while True:
            try:
                i, m = self.queue.get_nowait()
            except Empty:
                break

            self.setmessage(m, i)
            doref = True

        if doref:
            self.doprint()

    def setmessage(self, mess, i):
        self.message[i] = str(mess)

    def __del__(self):
        try:
            if self.have_curses:
                self.curses.echo()
                self.curses.endwin()
                print("\n" * (NTHREADS + 1))
        except Exception:
            pass

def format_time(secs):
    if secs < 60:
        return "%d s"%secs
    if secs < 60*3:
        return "%d m %d s"%divmod(secs,60)
    return "%d min"%(secs/60)

def showprogress(messtring,i,queue):
    global start_time

    elapsed_time = time.time() - start_time
    progress = float(itcounters[i])/(len(schedules[i])*NITER)

    try:
        ETA = elapsed_time / progress * (1-progress)
    except ZeroDivisionError:
        ETA = 0

    mes = "%d%%, %s remaining. Chunk %d/%d, %s"%(
            int(100*progress),
            format_time(ETA),
            chnkcounters[i],
            len(schedules[i]),
            messtring.ljust(30)
                )
    queue.put((i,mes))


#def showprogress(m,i):
#    pass

def raytrace_schedule(i,schedule,total_shared,q): # this is the function running on each thread
    #global schedules,itcounters,chnkcounters,killers

    if len(schedule) == 0:
        return

    total_colour_buffer_preproc = tonumpyarray(total_shared)

    #schedule = schedules[i]

    itcounters[i] = 0
    chnkcounters[i]= 0

    for chunk in schedule:
        #if killers[i]:
        #    break
        chnkcounters[i]+=1

        #number of chunk pixels
        numChunk = chunk.shape[0]

        #useful constant arrays
        ones = xp.ones((numChunk))
        ones3 = xp.ones((numChunk,3))
        UPFIELD = xp.outer(ones,xp.array([0.,1.,0.]))
        BLACK = xp.outer(ones,xp.array([0.,0.,0.]))

        #arrays of integer pixel coordinates
        x = chunk % RESOLUTION[0]
        y = chunk // RESOLUTION[0]

        showprogress("Generating view vectors...",i,q)

        #the view vector in 3D space
        view = xp.zeros((numChunk,3))

        view[:,0] = x.astype(float)/RESOLUTION[0] - .5
        view[:,1] = ((-y.astype(float)/RESOLUTION[1] + .5)*RESOLUTION[1])/RESOLUTION[0] #(inverting y coordinate)
        view[:,2] = 1.0

        view[:,0]*=TANFOV
        view[:,1]*=TANFOV

        #rotating through the view matrix

        view = xp.einsum('jk,ik->ij',viewMatrix,view)

        #original position
        point = xp.outer(ones, CAMERA_POS)

        normview = normalize(view)

        velocity = np.copy(normview)


        # initializing the colour buffer
        object_colour = xp.zeros((numChunk,3))
        object_alpha = xp.zeros(numChunk)

        #squared angular momentum per unit mass (in the "Newtonian fantasy")
        #h2 = xp.outer(sqrnorm(xp.cross(point,velocity)),xp.array([1.,1.,1.]))
        h2 = sqrnorm(xp.cross(point,velocity))[:,np.newaxis]

        pointsqr = np.copy(ones3)

        for it in range(NITER):
            itcounters[i]+=1

            if it%150 == 1:
                if killers[i]:
                    break
                showprogress("Raytracing...",i,q)

            # STEPPING
            oldpoint = np.copy(point) #not needed for tracing. Useful for intersections

            if METHOD == METH_LEAPFROG:
                #leapfrog method here feels good
                point += velocity * STEP

                if DISTORT:
                    #this is the magical - 3/2 r^(-5) potential...
                    accel = - 1.5 * h2 *  point / xp.power(sqrnorm(point),2.5)[:,np.newaxis]
                    velocity += accel * STEP

            elif METHOD == METH_RK4:
                if DISTORT:
                    #simple step size control
                    rkstep = STEP

                    # standard Runge-Kutta
                    y = xp.zeros((numChunk,6))
                    y[:,0:3] = point
                    y[:,3:6] = velocity
                    k1 = RK4f( y, h2)
                    k2 = RK4f( y + 0.5*rkstep*k1, h2)
                    k3 = RK4f( y + 0.5*rkstep*k2, h2)
                    k4 = RK4f( y +     rkstep*k3, h2)

                    increment = rkstep/6. * (k1 + 2*k2 + 2*k3 + k4)

                    velocity += increment[:,3:6]

                point += increment[:,0:3]


            #useful precalcs
            pointsqr = sqrnorm(point)
            #phi = xp.arctan2(point[:,0],point[:,2])    #too heavy. Better an instance wherever it's needed.
            #normvel = normalize(velocity)              #never used! BAD BAD BAD!!


            # FOG

            if FOGDO and (it%FOGSKIP == 0):
                phsphtaper = xp.clip(0.8*(pointsqr - 1.0),0.,1.0)
                fogint = xp.clip(FOGMULT * FOGSKIP * STEP / pointsqr,0.0,1.0) * phsphtaper
                fogcol = ones3

                object_colour = blendcolors(fogcol,fogint,object_colour,object_alpha)
                object_alpha = blendalpha(fogint, object_alpha)


            # CHECK COLLISIONS
            # accretion disk

            if DISK_TEXTURE_INT != DT_NONE:

                mask_crossing = xp.logical_xor( oldpoint[:,1] > 0., point[:,1] > 0.) #whether it just crossed the horizontal plane
                mask_distance = xp.logical_and((pointsqr < DISKOUTERSQR), (pointsqr > DISKINNERSQR))  #whether it's close enough

                diskmask = xp.logical_and(mask_crossing,mask_distance)

                if (diskmask.any()):

                    #actual collision point by intersection
                    lambdaa = - point[:,1]/velocity[:,1]
                    colpoint = point + lambdaa[:,np.newaxis] * velocity
                    colpointsqr = sqrnorm(colpoint)

                    if DISK_TEXTURE_INT == DT_GRID:
                        phi = xp.arctan2(colpoint[:,0],point[:,2])
                        theta = xp.arctan2(colpoint[:,1],norm(colpoint[:,[0,2]]))
                        diskcolor =     xp.outer(
                                xp.mod(phi,0.52359) < 0.261799,
                                            xp.array([1.,1.,0.])
                                                ) +  \
                                        xp.outer(ones,xp.array([0.,0.,1.]) )
                        diskalpha = diskmask

                    elif DISK_TEXTURE_INT == DT_SOLID:
                        diskcolor = xp.array([1.,1.,.98])
                        diskalpha = diskmask

                    elif DISK_TEXTURE_INT == DT_TEXTURE:

                        phi = xp.arctan2(colpoint[:,0],point[:,2])

                        uv = xp.zeros((numChunk,2))

                        uv[:,0] = ((phi+2*np.pi)%(2*np.pi))/(2*np.pi)
                        uv[:,1] = (xp.sqrt(colpointsqr)-DISKINNER)/(DISKOUTER-DISKINNER)

                        diskcolor = lookup ( texarr_disk, xp.clip(uv,0.,1.))
                        #alphamask = (2.0*ransample) < sqrnorm(diskcolor)
                        #diskmask = xp.logical_and(diskmask, alphamask )
                        diskalpha = diskmask * xp.clip(sqrnorm(diskcolor)/3.0,0.0,1.0)

                    elif DISK_TEXTURE_INT == DT_BLACKBODY:

                        temperature = xp.exp(bb.disktemp(colpointsqr,9.2103))

                        if REDSHIFT:
                            R = xp.sqrt(colpointsqr)

                            disc_velocity = 0.70710678 * \
                                        xp.power((xp.sqrt(colpointsqr)-1.).clip(0.1),-.5)[:,np.newaxis] * \
                                        xp.cross(UPFIELD, normalize(colpoint))


                            gamma =  xp.power( 1 - sqrnorm(disc_velocity).clip(max=.99), -.5)

                            # opz = 1 + z
                            opz_doppler = gamma * ( 1. + xp.einsum('ij,ij->i',disc_velocity,normalize(velocity)))
                            opz_gravitational = xp.power(
                                xp.maximum(1. - 1. / R, 1e-12),
                                -0.5
                                )

                            # (1+z)-redshifted Planck spectrum is still Planckian at temperature T
                            temperature /= (opz_doppler*opz_gravitational).clip(0.1)

                        intensity = bb.intensity(temperature)
                        if DISK_INTENSITY_DO:
                            diskcolor = xp.einsum('ij,i->ij', bb.colour(temperature),DISK_MULTIPLIER*intensity)#xp.maximum(1.*ones,DISK_MULTIPLIER*intensity))
                        else:
                            diskcolor = bb.colour(temperature)

                        diskalpha = xp.clip(
                            diskmask * DISK_ALPHA_MULTIPLIER * intensity,
                            0.,
                            1.
                            )

                    elif DISK_TEXTURE_INT == DT_HYBRID:

                        # Blackbody provides the physical colour and brightness
                        temperature = xp.exp(bb.disktemp(colpointsqr,9.2103))

                        if REDSHIFT:
                            R = xp.sqrt(colpointsqr)

                            disc_velocity = 0.70710678 * \
                                        xp.power((xp.sqrt(colpointsqr)-1.).clip(0.1),-.5)[:,np.newaxis] * \
                                        xp.cross(UPFIELD, normalize(colpoint))

                            gamma = xp.power(
                                1 - sqrnorm(disc_velocity).clip(max=.99),
                                -.5
                            )

                            opz_doppler = gamma * (
                                1. + xp.einsum(
                                    'ij,ij->i',
                                    disc_velocity,
                                    normalize(velocity)
                                )
                            )

                            opz_gravitational = xp.power(
                                xp.maximum(1. - 1. / R, 1e-12),
                                -0.5
                            )

                            temperature /= (
                                opz_doppler * opz_gravitational
                            ).clip(0.1)

                        intensity = bb.intensity(temperature)

                        if DISK_INTENSITY_DO:
                            diskcolor = xp.einsum(
                                'ij,i->ij',
                                bb.colour(temperature),
                                DISK_MULTIPLIER * intensity
                            )
                        else:
                            diskcolor = bb.colour(temperature)

                        # Get the same texture coordinates used by the
                        # normal texture disk renderer.
                        phi = xp.arctan2(colpoint[:,0],point[:,2])

                        uv = xp.zeros((numChunk,2))

                        uv[:,0] = ((phi+2*np.pi)%(2*np.pi))/(2*np.pi)
                        uv[:,1] = (
                            (xp.sqrt(colpointsqr)-DISKINNER)
                            /(DISKOUTER-DISKINNER)
                        )

                        disktexture = lookup(
                            texarr_disk,
                            xp.clip(uv,0.,1.)
                        )

                        ## Texture controls the local structure of the blackbody disk.
                        # Convert RGB texture to luminance.
                        texture_brightness = (
                            0.2126 * disktexture[:,0] +
                            0.7152 * disktexture[:,1] +
                            0.0722 * disktexture[:,2]
                        )

                        # Increase contrast so the structure is actually visible.
                        texture_brightness = xp.clip(
                            (texture_brightness - 0.5) * 2.0 + 0.5,
                            0.0,
                            1.0
                        )

                        # Modulate the blackbody emission with the texture.
                        texture_factor = 0.35 + 0.65 * texture_brightness

                        diskcolor *= texture_factor[:,np.newaxis]

                        # Blackbody controls the physical opacity.
                        diskalpha = xp.clip(
                            diskmask * DISK_ALPHA_MULTIPLIER * intensity,
                            0.,
                            1.
                        )


                    object_colour = blendcolors(diskcolor,diskalpha,object_colour,object_alpha)
                    object_alpha = blendalpha(diskalpha, object_alpha)



            # event horizon
            oldpointsqr = sqrnorm(oldpoint)

            mask_horizon = xp.logical_and((pointsqr < 1),(sqrnorm(oldpoint) > 1) )

            if mask_horizon.any() :

                lambdaa =  1. - ((1.-oldpointsqr)/((pointsqr - oldpointsqr)))[:,np.newaxis]
                colpoint = lambdaa * point + (1-lambdaa)*oldpoint

                if HORIZON_GRID:
                    phi = xp.arctan2(colpoint[:,0],point[:,2])
                    theta = xp.arctan2(colpoint[:,1],norm(point[:,[0,2]]))
                    horizoncolour = xp.outer( xp.logical_xor(xp.mod(phi,1.04719) < 0.52359,xp.mod(theta,1.04719) < 0.52359), xp.array([1.,0.,0.]))
                else:
                    horizoncolour = BLACK#xp.zeros((numPixels,3))

                horizonalpha = mask_horizon

                object_colour = blendcolors(horizoncolour,horizonalpha,object_colour,object_alpha)
                object_alpha = blendalpha(horizonalpha, object_alpha)



        showprogress("generating sky layer...",i,q)

        vphi = xp.arctan2(velocity[:,0],velocity[:,2])
        vtheta = xp.arctan2(velocity[:,1],norm(velocity[:,[0,2]]) )

        vuv = xp.zeros((numChunk,2))

        vuv[:,0] = xp.mod(vphi+4.5,2*np.pi)/(2*np.pi)
        vuv[:,1] = (vtheta+np.pi/2)/(np.pi)

        if SKY_TEXTURE_INT == DT_TEXTURE:
            col_sky = lookup(texarr_sky,vuv)[:,0:3]

        showprogress("generating debug layers...",i,q)

        ##debug color: direction of view vector
        #dbg_viewvec = xp.clip(view + vec3(.5,.5,0.0),0.0,1.0)
        ##debug color: direction of final ray
        ##debug color: grid
        #dbg_grid = xp.abs(normalize(velocity)) < 0.1


        if SKY_TEXTURE_INT == ST_TEXTURE:
            col_bg = col_sky
        elif SKY_TEXTURE_INT == ST_NONE:
            col_bg = xp.zeros((numChunk,3))
        elif SKY_TEXTURE_INT == ST_FINAL:
            dbg_finvec = xp.clip(normalize(velocity) + xp.array([.5,.5,0.0])[np.newaxis,:],0.0,1.0)
            col_bg = dbg_finvec
        else:
            col_bg = xp.zeros((numChunk,3))


        showprogress("blending layers...",i,q)

        col_bg_and_obj = blendcolors(SKYDISK_RATIO*col_bg, ones ,object_colour,object_alpha)

        showprogress("beaming back to mothership.",i,q)
        # copy back in the buffer
        if not DISABLE_SHUFFLING:
            total_colour_buffer_preproc[chunk] = col_bg_and_obj
        else:
            total_colour_buffer_preproc[chunk[0]:(chunk[-1]+1)] = col_bg_and_obj


        #refresh display
        # NO: plt does not allow drawing outside main thread
        #if not DISABLE_DISPLAY:
        #    showprogress("updating display...")
        #    plt.imshow(total_colour_buffer_preproc.reshape((RESOLUTION[1],RESOLUTION[0],3)))
        #    plt.draw()

        showprogress("garbage collection...",i,q)
        gc.collect()

    showprogress("Done.",i,q)

# Threading

# setup
# Threading

output = Outputter()

process_list = []
for i in range(NTHREADS):
    p = multi.Process(
        target=raytrace_schedule,
        args=(i, schedules[i], total_colour_buffer_preproc_shared, output.queue)
    )
    process_list.append(p)

logger.debug("Starting threads...")

output.start_display()

for proc in process_list:
    proc.start()

try:
    refreshcounter = 0
    while True:
        refreshcounter += 1
        time.sleep(0.1)

        output.parsemessages()

        if not DISABLE_DISPLAY and (refreshcounter % 40 == 0):
            output.setmessage("Updating display...", -1)
            plt.imshow(total_colour_buffer_preproc.reshape((RESOLUTION[1], RESOLUTION[0], 3)))
            plt.draw()

        output.setmessage("Idle.", -1)

        alldone = True
        for i in range(NTHREADS):
            if process_list[i].is_alive():
                alldone = False

        if alldone:
            break

except KeyboardInterrupt:
    for i in range(NTHREADS):
        killers[i] = True
    sys.exit()

output.stop_display()
del output

logger.debug("Done tracing.")

logger.debug("Total raytracing time: %s", datetime.timedelta(seconds=(time.time() - start_time)))


logger.debug("Postprocessing...")

#gain
logger.debug("- gain...")
total_colour_buffer_preproc *= GAIN


# airy bloom
if AIRY_BLOOM:

    logger.debug("-computing Airy disk bloom...")

    #blending bloom

    #colour = total_colour_buffer_preproc + 0.3*blurd #0.2*dbg_grid + 0.8*dbg_finvec

    #airy disk bloom

    colour_bloomd = np.copy(total_colour_buffer_preproc)
    colour_bloomd = colour_bloomd.reshape((RESOLUTION[1],RESOLUTION[0],3))


    # the float constant is 1.22 * 650nm / (4 mm), the typical diffractive resolution
    # of the human eye for red light. It's in radians, so we rescale using field of view.
    radd = 0.00019825 * RESOLUTION[0] / xp.arctan(TANFOV)

    # the user is allowed to rescale the resolution, though
    radd*=AIRY_RADIUS

    # the pixel size of the kernel:
    # 25 pixels radius is ok for 5.0 bright source pixel at 1920x1080, so...
    # remembering that airy ~ 1/x^3, so if we want intensity/x^3 < hreshold =>
    # => max_x = (intensity/threshold)^1/3
    # so it scales with
    # - the cube root of maximum intensity
    # - linear in resolution

    mxint = xp.amax(colour_bloomd)

    kern_radius = 25 * xp.power( xp.amax(colour_bloomd) / 5.0 , 1./3.) * RESOLUTION[0]/1920.

    logger.debug("--(radius: %3f, kernel pixel radius: %3f, maximum source brightness: %3f)", radd, kern_radius, mxint)

    colour_bloomd = bloom.airy_convolve(colour_bloomd,radd)

    colour_bloomd = colour_bloomd.reshape((numPixels,3))


    colour_pb = colour_bloomd
else:
    colour_pb = total_colour_buffer_preproc


# wide gaussian (lighting dust effect)

if BLURDO:

    logger.debug("-computing wide gaussian blur...")

    #hipass = xp.outer(sqrnorm(total_colour_buffer_preproc) > BLOOMCUT, xp.array([1.,1.,1.])) * total_colour_buffer_preproc
    blurd = np.copy(total_colour_buffer_preproc)

    blurd = blurd.reshape((RESOLUTION[1],RESOLUTION[0],3))

    for i in range(2):
        logger.debug("- gaussian blur pass %d...", i)
        blurd = ndim.gaussian_filter(blurd,int(0.05*RESOLUTION[0]))

    blurd = blurd.reshape((numPixels,3))
    colour = colour_pb + 0.2 * blurd
else:
    colour = colour_pb



#normalization
if NORMALIZE > 0:
    logger.debug("- normalizing...")
    colour *= 1 / (NORMALIZE * xp.amax(colour.flatten()) )




#final colour
colour = xp.clip(colour,0.,1.)


logger.debug("Conversion to image and saving...")

saveToImg(colour,"tests/out.png")
saveToImg(total_colour_buffer_preproc,"tests/preproc.png")
if BLURDO:
    saveToImg(colour_pb,"tests/postbloom.png")
