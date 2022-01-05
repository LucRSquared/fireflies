from colorsys import rgb_to_hsv, hsv_to_rgb
from math import floor, sqrt, inf
import scipy.spatial
import pandas as pd
import numpy as np
from tqdm import tqdm

RGB_YELLOWISH_GREEN = (243, 243, 97)

rng = np.random.default_rng(2022) # Random Number Generator

def set_rng_seed(rng,seed):
    rng = np.random.default_rng(seed)
    return rng

def normalize_rgb(rgb255) -> tuple:
    # Converts rgb tuple with values 0 to 255 into 0 to 1
    return tuple([x/float(255) for x in rgb255])
    
def denormalize_rgb(rgb01) -> tuple:
    # Converts rgb tuple with values 0 to 1 into 0 to 255
    return tuple([floor(x*float(255)) for x in rgb01])

def change_color_intensity(newintensity, rgb=RGB_YELLOWISH_GREEN) -> tuple:
    rgb01 = normalize_rgb(rgb)
    hsv = list(rgb_to_hsv(rgb01[0],rgb01[1],rgb01[2]))
    oldintensity = hsv[2]
    hsv[2] = newintensity*oldintensity
    rgbnew = hsv_to_rgb(hsv[0],hsv[1],hsv[2])
    rgbnew = denormalize_rgb(rgbnew)
    return rgbnew

def generate_christmastree_csv_file(df,outputfile='fireflies.csv',rgbmaxcolor=RGB_YELLOWISH_GREEN):
    N = df['firefly_id'].max()+1 # +1 to get the actual number of fireflies

    FrameIdMax = df['frame_id'].max()

    with open(outputfile,'w') as f:
        firstline = [f'R_{i},G_{i},B_{i}' for i in range(N)]
        firstline.insert(0,'FRAME_ID')
        firstline = ','.join(firstline)+'\n'

        f.write(firstline)
        all_rows = []
        
        for i in tqdm(range(FrameIdMax)):
            therow = [val for ffid in range(N) 
                        for val in change_color_intensity(df[(df['frame_id']==i) & (df['firefly_id']==ffid)]['intensity'].values[0],
                                                                rgb=rgbmaxcolor)]
            therow.insert(0,int(i))
            therow = [str(number) for number in therow]
            therow = ','.join(therow)+'\n'
            all_rows.append(therow)
            # f.write(therow)
        
        f.writelines(all_rows)

    return

class Firefly:
    # The class for one firefly

    def __init__(self, id=None, x=[], y=[], z=[],
                    clock=0, clocklength=100, clockratiothreshold=0.25,
                    ratioclocknudgelength=0.1, fovratio=0.4, visiondistance=inf, 
                    intensityperception=0.9,incrementstep=1.1,
                    rgbcolormax=RGB_YELLOWISH_GREEN) -> None:
        
        self.id = id
        self.x  = x
        self.y  = y
        self.z  = z
        
        self.clock = clock # current internal clock value
        self.clocklength = clocklength # frame duration of the firefly cycle
        
        self.ratioclocknudgelength = ratioclocknudgelength
        self.clockratiothreshold = clockratiothreshold # controls the pulse intensity fading
        self.incrementstep = incrementstep
        self.rgbcolormax = rgbcolormax # color at its most luminosity
        self.intensity = []
        self.update_intensity() # initial value of intensity based on clock
        self.rgbcolorcurrent = change_color_intensity(self.intensity, rgbcolormax)


        self.fovratio = fovratio # fraction of total 360 visible by firefly
        self.visiondistance = visiondistance # distance at which firefly can "see"
        self.intensityperception = intensityperception # threshold above which intensity can be seen

    def randomize_clock(self): 
        self.clock = float(rng.lognormal(mean=self.clocklength/5, sigma=0.1*self.clocklength,size=1)[0])
        # self.clock = float(rng.exponential(scale=self.clocklength/3, size=1)[0])
        if self.clock >= self.clocklength:
            self.clock = 0
        self.update_intensity()
        return

    def increment_clock(self):
        self.clock = self.clock+self.incrementstep
        if self.clock >= self.clocklength:
            self.clock = 0
        return

    def update_intensity(self) -> None:
        # waning pulse------w*clocklength-------(1-w)*clocklength-----increasing pulse
        w = self.clockratiothreshold
        c = self.clock
        cmax = self.clocklength

        assert 0 < w <= 0.5

        if w*cmax < c < (1.0-w)*cmax:
            self.intensity = 0
        elif 0.0 <= c <= w*cmax:
            self.intensity = 1.0 - c/(w*cmax)
        else:
            self.intensity = (1.0/(w*cmax))*(c - (1.0-w)*cmax)
        return

    def update_color(self):
        self.rgbcolorcurrent = change_color_intensity(self.intensity, self.rgbcolormax)
        return

    def update_light(self):
        self.update_intensity()
        self.update_color()

    def distance_from_other(self, other) -> float:
        return sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)

    def i_see_you(self, other, indistance=None) -> bool:
        r = rng.uniform(low=0, high=1, size=1)[0] 
        # randomly decide if the firefly has the correct orientation
        # compared to field of view
        # indistance can be manually provided if the other is in visible distance and
        # has already been checked to avoid checking every time
        if indistance == None:
            if (self.distance_from_other(self, other) <= self.visiondistance) \
                                            and (r <= self.fovratio) \
                                            and (other.intensity >= self.intensityperception):
                return True
            else:
                return False
        elif (indistance == True) and (r <= self.fovratio) \
             and (other.intensity >= self.intensityperception):
            return True
        else:
            return False

    def you_see_me(self, other, indistance=None) -> bool:
        r = rng.uniform(low=0, high=1, size=1)[0] 
        # randomly decide if the other firefly has the correct orientation
        # compared to field of view
        # indistance can be manually provided if the other is in visible distance and
        # has already been checked to avoid checking every time
        if indistance == None:
            if (self.distance_from_other(self, other) <= self.visiondistance) \
                                            and (r <= self.fovratio) \
                                            and (self.intensity >= other.intensityperception):
                return True
            else:
                return False
        elif (indistance == True) and (r <= other.fovratio) \
             and (self.intensity >= other.intensityperception):
            return True
        else:
            return False

    def nudge_clock(self):
        chaos=1 #rng.uniform(0.99,1.01,1)[0]
        self.clock = self.clock + chaos*self.ratioclocknudgelength*self.clocklength
        if self.clock >= self.clocklength:
            self.clock = 0
        return True

    def update_position(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        return 


class Swarm:
    # Collection of N fireflies (Firefly class)
    
    def __init__(self, visibledistance=inf, rememberiterations=False, incrementstep=1.1):
        ''' Upon initialization, an empty swarm is created
            Use the "change" functions to use non-default values
        '''
        self.instancesdict = {}
        # self.instancesdict = weakref.WeakValueDictionary()

        self.N = 0 # Number of fireflies
        self.positions = [] # df of positions
        self.intensities = [] # array of intensities
        self.clocks = [] # array of clocks
        self.visibledistance = visibledistance
        self.incrementstep = incrementstep
        self.distances = [] # matrix of distances of firefly to one another
        self.iisinvisibledistanceofj = [] # matrix that says if two fireflies are close enough
                                      # to be visible by one another 
        self.visibilityofjtoi = [] # visibility matrix taking into account
                             # distance + firefly orientation + intensity of neighbors
        self.rememberiterations = rememberiterations

        self.intensities_history = []
        self.clocks_history = []
        self.frames_history = []
        self.fireflies_history = []
        self.X_history = []
        self.Y_history = []
        self.Z_history = []
        self.frameid = 0


        return

    def read_positions_from_csv(self, fullfilepath):
        self.positions = pd.read_csv(fullfilepath, names=("X", "Y", "Z"))
        self.N = self.population(updateswarm=True)
        return

    def generate_N_random_positions_in_cube(self,N,xrange=[0,1],yrange=[0,1],zrange=[0,1]):
        self.N = N
        X = rng.uniform(low=xrange[0],high=xrange[1],size=N)
        Y = rng.uniform(low=yrange[0],high=yrange[1],size=N)
        Z = rng.uniform(low=zrange[0],high=zrange[1],size=N)

        self.positions = pd.DataFrame({'X':X,'Y':Y,'Z':Z})

        return

    def population(self,updateswarm=False):
        N = len(self.positions)
        if N!=0:
            N = self.positions.shape[0]

        if updateswarm:
            self.N = N

        return N

    def update_firefly_position_in_global_array(self, id):
        N = self.N
        if id > N or id < 0:
            print(f"You're trying to update id = {id} the number should be \
                    between 0 and {N}. Nothing happened")
        elif ~isinstance(id, int):
            print(f"You entered id = {id}, it should be an int. Nothing happened")
        else:
            self.positions['X'][id] = self.instances[id].x
            self.positions['Y'][id] = self.instances[id].y
            self.positions['Z'][id] = self.instances[id].z
        return

    def update_all_firefly_positions_in_global_array(self):
        N = self.N
        for id in range(N):
            self.update_firefly_position_in_global_array(id)
        return

    def update_distance_matrix(self):
        arr = scipy.spatial.distance.pdist(np.array(self.positions))
        self.distances = scipy.spatial.distance.squareform(arr)
        return

    def update_visibility_distance_matrix(self):
        # self.iisinvisibledistanceofj = (self.distances.std()/20 <= self.distances) & (self.distances <= self.visibledistance)
        self.iisinvisibledistanceofj = self.distances <= self.visibledistance
        return

    def update_visibility_matrix(self): #TODO: add possibility to compute everything without using distance matrix
        N = self.N
        self.visibilityofjtoi = np.full((N,N), False)
        for i in range(N):
            idx_fireflies_in_range = np.where(self.iisinvisibledistanceofj[i,:] == True)[0]
            for j in idx_fireflies_in_range:
                if j == i: # do i see myself?
                    self.visibilityofjtoi[i,j] = False
                elif self.visibilityofjtoi[j,i] == False: # NOTE the inversion of the indexes, we skip if it's
                                                          # already determined to be positive
                    # self.visibilityofjtoi[j,i] = self.instancesdict[i].you_see_me(self.instancesdict[j],True)
                    if self.instancesdict[i].you_see_me(self.instancesdict[j],True):
                        self.visibilityofjtoi[j,i] = True
                        # self.instancesdict[j].nudge_clock()
                        
                
        return

    def update_intensities_array(self):
        N = self.N
        self.intensities = []
        for i in range(N):
            self.intensities.append(self.instancesdict[i].intensity)
        return

    def update_clocks_array(self):
        N = self.N
        self.clocks = []
        for i in range(N):
            self.clocks.append(self.instancesdict[i].clock)
        return

    def randomize_all_clocks(self):
        N = self.N
        for i in range(N):
            self.instancesdict[i].randomize_clock()
        return

    def half_and_half_clocks(self,clockhigh,clocklow=0):

        zmean = self.positions['Z'].mean()
        N = self.N

        for i in range(N):
            if self.positions['Z'].iloc[i] <= zmean:
                self.instancesdict[i].clock = float(clocklow)
            else:
                self.instancesdict[i].clock = float(clockhigh)
        return

    def increment_all_clocks(self):
        N = self.N
        for i in range(N):
            self.instancesdict[i].increment_clock()
        return
        
    def change_clocklengths(self,clocklengths):
        # if clocklengths is array of size N then each firefly clocklengths is changed
        # if scalar then same number assigned to all
        N = self.N
        if isinstance(clocklengths, int) or isinstance(clocklengths, float):
            clocklengths = np.full((N,1),clocklengths)
        elif len(clocklengths) != N:
            print(f"clocklengths is of size {len(clocklengths)}, it should be 1 or {N}")
        elif len(clocklengths) == N:
            pass # ok
        
        for i in range(N):
            self.instancesdict[i].clocklength = float(clocklengths[i])
        return

    def change_clocks(self, clocks):
        # if clocks is array of size N then each firefly clock is changed
        # if scalar then same number assigned to all
        N = self.N
        if isinstance(clocks, int) or isinstance(clocks, float):
            clocks = np.full((N,1),clocks)
        elif len(clocks) != N:
            print(f"clocks is of size {len(clocks)}, it should be 1 or {N}")
            return
        elif len(clocks) == N:
            pass # ok
            
        for i in range(N):
            self.instancesdict[i].clock = float(clocks[i])
        return

    def change_clockratiothresholds(self, clockratiothresholds):
        # if clockratiothresholds is array of size N then 
        # each firefly clockratiothreshold is changed
        # if scalar then same number assigned to all
        N = self.N
        if isinstance(clockratiothresholds, float) or clockratiothresholds == 1:
            clockratiothresholds = np.full((N,1),clockratiothresholds)
        elif len(clockratiothresholds) != N:
            print(f"clockratiothresholds is of size {len(clockratiothresholds)},\
                    it should be 1 or {N}")
            return
        elif len(clockratiothresholds) == N:
            pass # ok
            
        for i in range(N):
            self.instancesdict[i].clockratiothreshold = float(clockratiothresholds[i])
        return

    def change_ratioclocknudgelengths(self, ratioclocknudgelengths):
        # if ratioclocknudgelengths is array of size N then 
        # each firefly ratioclocknudgelength is changed
        # if scalar then same number assigned to all
        N = self.N
        if isinstance(ratioclocknudgelengths, float) or ratioclocknudgelengths == 1:
            ratioclocknudgelengths = np.full((N,1),ratioclocknudgelengths)
        elif len(ratioclocknudgelengths) != N:
            print(f"ratioclocknudgelengths is of size {len(ratioclocknudgelengths)},\
                    it should be 1 or {N}")
            return
        elif len(ratioclocknudgelengths) == N:
            pass # ok
            
        for i in range(N):
            self.instancesdict[i].ratioclocknudgelength = float(ratioclocknudgelengths[i])
        return

    def change_fovratios(self, fovratios):
        # if fovratios is array of size N then 
        # each firefly fovratio is changed
        # if scalar then same number assigned to all
        N = self.N
        if isinstance(fovratios, float) or fovratios == 1:
            fovratios = np.full((N,1),fovratios)
        elif len(fovratios) != N:
            print(f"fovratios is of size {len(fovratios)},\
                    it should be 1 or {N}")
            return
        elif len(fovratios) == N:
            pass # ok
            
        for i in range(N):
            self.instancesdict[i].fovratio = float(fovratios[i])
        return

    def change_intensityperceptions(self, intensityperceptions):
        # if intensityperceptions is array of size N then 
        # each firefly intensityperception is changed
        # if scalar then same number assigned to all
        N = self.N
        if isinstance(intensityperceptions, float) or intensityperceptions == 1:
            intensityperceptions = np.full((N,1),intensityperceptions)
        elif len(intensityperceptions) != N:
            print(f"intensityperceptions is of size {len(intensityperceptions)},\
                    it should be 1 or {N}")
            return
        elif len(intensityperceptions) == N:
            pass # ok
            
        for i in range(N):
            self.instancesdict[i].intensityperception = float(intensityperceptions[i])
        return

    def change_rgbcolormaxs(self, rgbcolormaxs):
        # if rgbcolormaxs is array of tuples (255,255,255) of size N then 
        # each firefly intensityperception is changed
        # if scalar then same number assigned to all
        N = self.N
        if isinstance(rgbcolormaxs, tuple):
            rgbcolormaxs = [rgbcolormaxs for _ in range(N)]
        elif len(rgbcolormaxs) != N:
            print(f"rgbcolormaxs is of size {len(rgbcolormaxs)},\
                    it should be 1 or {N}")
            return
        elif len(rgbcolormaxs) == N:
            pass # ok
            
        for i in range(N):
            self.instancesdict[i].rgbcolormax = rgbcolormaxs[i]
        return

    def change_firefly_xyz(self,id,xyzlist):
        self.instances[id].X = xyzlist[1]
        self.instances[id].Y = xyzlist[2]
        self.instances[id].Z = xyzlist[3]
        self.update_firefly_position_in_global_array(id)
        return

    def change_visiondistance(self, visiondistance):
        #WARNING that only a scalar is used here not an array
        N = self.N
        self.visibledistance = visiondistance # change it for the Swarm
        if isinstance(visiondistance, float):            
            for i in range(N):
                self.instancesdict[i].visiondistance = visiondistance
        else:
            raise Exception('Please provide a float for visiondistance')
        return

    def change_incrementstep(self, incrementstep):
        #WARNING that only a scalar is used here not an array
        N = self.N
        self.incrementstep = incrementstep # change it for the Swarm
        if isinstance(incrementstep, float) or isinstance(incrementstep, int):            
            for i in range(N):
                self.instancesdict[i].incrementstep = float(incrementstep)
        else:
            raise Exception('Please provide a float for incrementstep')
        return

    def change_rememberiterations(self, rememberiterations):
        if isinstance(rememberiterations,bool):
            self.rememberiterations = rememberiterations
        else:
            raise Exception('Please provide an object of type bool for rememberiterations')
        return

    def change_parameters_with_dictionary(self,inputs):
        for key, value in inputs.items():
            if key == "clocklengths":
                self.change_clocklengths(value)
            elif key == "clocks":
                self.change_clocks(value)
            elif key == "clockratiothresholds":
                self.change_clockratiothresholds(value)
            elif key == "ratioclocknudgelengths":
                self.change_ratioclocknudgelengths(value)
            elif key == "fovratios":
                self.change_fovratios(value)
            elif key == "intensityperceptions":
                self.change_intensityperceptions(value)
            elif key == "rgbcolormaxs":
                self.change_rgbcolormaxs(value)
            elif key == "visiondistance":
                self.change_visiondistance(value)
            elif key == "incrementstep":
                self.change_incrementstep(value)
            elif key == "rememberiterations":
                self.change_rememberiterations(value)
            else:
                print(f"Invalid key = {key}")
        return

    def remember_clocks(self):
        self.clocks_history.extend(self.clocks)
        return

    def remember_intensities(self):
        self.intensities_history.extend(self.intensities)
        return

    def remember_frames(self):
        self.frames_history.extend([self.frameid]*self.N)
        return

    def remember_fireflies(self):
        self.fireflies_history.extend(list(range(self.N)))
        return

    def remember_positions(self):
        self.X_history.extend(list(self.positions['X']))
        self.Y_history.extend(list(self.positions['Y']))
        self.Z_history.extend(list(self.positions['Z']))
        pass

    def remember_all(self):
        self.remember_frames()
        self.remember_clocks()
        self.remember_intensities()
        self.remember_fireflies()
        self.remember_positions()
        return

    def generate_history_df(self):
        masterlist = [list(i) for i in zip(self.frames_history,
                                           self.fireflies_history,
                                           self.X_history,
                                           self.Y_history,
                                           self.Z_history,
                                           self.intensities_history,
                                           self.clocks_history)]
 
        df = pd.DataFrame(masterlist,columns=['frame_id','firefly_id','X','Y','Z','intensity','clock'])    
        return df
    
    def generate_instances_from_positions_with_default_parameters(self):
        N = self.N
        if N == 0:
            raise Exception('No instances generated')
        else:
            df = self.positions
            for i in range(N):
                self.instancesdict[i] = Firefly(id=i, x=df['X'][i], y=df['X'][i], z=df['X'][i])
        return
        
    def nudge_all_clocks(self):
        # Nudge all clocks according to visibilityofjtoi
        # if a firefly sees at least one other firefly then it's nudgle
        N = self.N
        fireflies_to_nudge = np.any(self.visibilityofjtoi,axis=0) # True if any j is visible to i
        fireflies_to_nudge_range = [i for i in range(len(fireflies_to_nudge)) if fireflies_to_nudge[i]==True]
        for i in fireflies_to_nudge_range:
            self.instancesdict[i].nudge_clock()
        return

    def update_all_lights(self):
        N = self.N
        for i in range(N):
            self.instancesdict[i].update_light()
        return

    def do_one_step(self):
        self.increment_all_clocks()
        self.update_visibility_matrix()        
        self.nudge_all_clocks()
        self.update_all_lights()
        self.update_intensities_array()
        self.update_clocks_array()
        # does not take into account that fireflies are moving
        # should have some "update position function here"
        if self.rememberiterations:
            self.remember_all() # remember what was just computed       
            # self.remember_all_no_XYZ() # remember what was just computed       
        return

    def increment_frameid(self):
        self.frameid += 1
        return
    