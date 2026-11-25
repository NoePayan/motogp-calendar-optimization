import unittest
import math
import os
import csv
import random
import copy
import numpy as np
import pyswarms as ps
from simanneal import Annealer
from deap import base, creator, tools, algorithms

# constants that define the likely hood of two individuals having crossover
# performed and the probability that a child will be mutated. needed for the
# DEAP library
CXPB = 0.5
MUTPB = 0.2

# base directory of this script, used to build absolute paths to the CSV
# files so the script works no matter what directory it is launched from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_LOCATIONS_FILE = os.path.join(BASE_DIR, 'track-locations.csv')
RACE_WEEKENDS_FILE = os.path.join(BASE_DIR, 'race-weekends.csv')

# define a spurious best solution and cost. We will update this as we update the cost of each particle
swarm_best_cost = 1000000
swarm_best_itinerary = []
itineraries = []
home = 11


# the unit tests to check that the simulation has been implemented correctly
class UnitTests(unittest.TestCase):
    # this will read in the track locations file and will pick out 5 fields to see if the file has been read correctly
    def testReadCSV(self):
        # read in the locations file
        rows = readCSVFile(TRACK_LOCATIONS_FILE)

        # test that the corners and a middle value are read in correctly
        self.assertEqual('GP', rows[0][0])
        self.assertEqual('Valencia', rows[0][22])
        self.assertEqual('Temp week 52', rows[55][0])
        self.assertEqual('16.25', rows[55][22])
        self.assertEqual('12.5', rows[11][8])

    # this will test to see if the row conversion works. here we will convert the latitude rwo and will test 5 values
    # as we are dealing with floating point we will use almost equals rather than a direct equality
    def testRowToFloat(self):
        # read in the locations file and convert the latitude column to floats
        rows = readCSVFile(TRACK_LOCATIONS_FILE)
        convertRowToFloat(rows, 2)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(14.957883, rows[2][1], delta=0.0001)
        self.assertAlmostEqual(39.484786, rows[2][22], delta=0.0001)
        self.assertAlmostEqual(36.532176, rows[2][17], delta=0.0001)
        self.assertAlmostEqual(-38.502284, rows[2][19], delta=0.0001)
        self.assertAlmostEqual(36.709896, rows[2][5], delta=0.0001)

        # check that the conversion of a temperature row to floating point is also correct
        convertRowToFloat(rows, 5)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(31.5, rows[5][1], delta=0.0001)
        self.assertAlmostEqual(16.5, rows[5][22], delta=0.0001)
        self.assertAlmostEqual(8.5, rows[5][17], delta=0.0001)
        self.assertAlmostEqual(23.5, rows[5][19], delta=0.0001)
        self.assertAlmostEqual(16.5, rows[5][5], delta=0.0001)

    # # this will test to see if the file conversion overall is successful for the track locations
    # # it will read in the file and will test a string, float, and int from 2 rows to verify it worked correctly
    def testReadTrackLocations(self):
        # read in the locations file
        rows = readTrackLocations()

        # check the name, latitude, and final temp of the first race
        self.assertEqual(rows[0][0], 'Thailand')
        self.assertAlmostEqual(rows[2][0], 14.957883, delta=0.0001)
        self.assertAlmostEqual(rows[55][0], 30.75, delta=0.0001)

        # check the name, longitude, and initial temp of the last race
        self.assertEqual(rows[0][21], 'Valencia')
        self.assertAlmostEqual(rows[2][21], 39.484786, delta=0.0001)
        self.assertAlmostEqual(rows[4][21], 16, delta=0.0001)

    # # tests to see if the race weekends file is read in correctly
    def testReadRaceWeekends(self):
        # read in the race weekends file
        weekends = readRaceWeekends()

        # check that thailand is weekend 8 and valencia is weekend 45
        self.assertEqual(weekends[0], 8)
        self.assertEqual(weekends[21], 45)

        # check that Austria is weekend 32
        self.assertEqual(weekends[12], 32)

    # # this will test to see if the haversine function will work correctly we will test 4 sets of locations
    def testHaversine(self):
        # read in the locations file with conversion
        rows = readTrackLocations()

        # check the distance of Thailand against itself this should be zero
        self.assertAlmostEqual(haversine(rows, 0, 0), 0.0, delta=0.01)

        # check the distance of Thailand against Silverstone this should be 9632.57 km
        self.assertAlmostEqual(haversine(rows, 0, 6), 9632.57, delta=0.01)

        # check the distance of silverstone against mugello this should be 1283.1 Km
        self.assertAlmostEqual(haversine(rows, 6, 8), 1283.12, delta=0.01)

        # check the distance of mugello to the red bull ring this should be 445.06 Km
        self.assertAlmostEqual(haversine(rows, 8, 12), 445.06, delta=0.01)

    # # will test to see if the season distance calculation is correct using the 2025 calendar
    def testDistanceCalculation(self):
        # read in the locations & race weekends, generate the weekends, and calculate the season distance
        tracks = readTrackLocations()
        weekends = readRaceWeekends()

        # calculate the season distance using mugello as the home track as this will be the case for almost all of the teams we will use silverstone for the others
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 8), 146768.1778, delta=0.0001)
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 6), 151481.2754, delta=0.0001)

    # # will test that the temperature constraint is working this should fail as azerbijan should fail the test
    def testTempConstraint(self):
        # load in the tracks, race weekends, and the sundays
        tracks = readTrackLocations()
        weekends1 = [8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 28, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45]
        weekends2 = [8, 10, 12, 14, 16, 18, 30, 22, 24, 25, 27, 28, 32, 33, 35, 36, 38, 39, 48, 42, 40, 41]

        # the test with the default calender should be false because of Great Britian at 17.25
        self.assertEqual(checkTemperatureConstraint(tracks, weekends1, 20, 35), False)
        self.assertEqual(checkTemperatureConstraint(tracks, weekends2, 20, 35), True)

    # # will test that we can detect a period for a summer shutdown in the prescribed weeks
    def testSummerShutdown(self):
        # weekend patterns the first has a summer shutdown the second doesn't
        weekends1 = [8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 28, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45]
        weekends2 = [8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 28, 31, 33, 35, 36, 38, 39, 41, 42, 44, 45]

        # the first should pass and the second should fail
        self.assertEqual(checkSummerShutdown(weekends1), True)
        self.assertEqual(checkSummerShutdown(weekends2), False)


# implementation of an annealer that will attempt to come up with a better calendar. it will allow the free movement of weekends with the exception of
# monaco. bahrain and abu dhabi will open and close the season respectively. the summer shutdown should still be respected. Double and triple header weekends
# are permitted but for or more races in a row is not allowed.
class CalendarAnnealer(Annealer):
    # used to initialise the annealer the state here will be the list of race weekends and the locations. we take in all the extra parameters
    # to help calculate the distance and temperature requirements on the way
    def __init__(self, weekends, home, tracks):
        self.home = home
        self.tracks = tracks
        super(CalendarAnnealer, self).__init__(weekends)

    # used to make a move this will take two race weekends and will swap their locations as this is the smallest change we can make
    def move(self):
        #checks if there is a temperature lower than the minimum
        MinIndex = indexLowestTemp(self.tracks, self.state, 15)
        if MinIndex != -1:
            Valid = [
                i for i in range(len(self.state))
                if i != 21 and i != MinIndex
            ]
            SwapChoice = random.choice(Valid)
            self.state[MinIndex], self.state[SwapChoice] = (
                self.state[SwapChoice],
                self.state[MinIndex],
            )
            return
        #if not checks if there is a temperature higher than the maximum
        MaxIndex = indexHighestTemp(self.tracks, self.state, 35)
        if MaxIndex != -1:
            Valid = [
                i for i in range(len(self.state))
                if i != 21 and i != MaxIndex
            ]
            SwapChoice = random.choice(Valid)
            self.state[MaxIndex], self.state[SwapChoice] = (
                self.state[SwapChoice],
                self.state[MaxIndex],
            )
            return

        # otherwise swaps any race EXCEPT valencia
        Valid = [i for i in range(len(self.state)) if i != 21]
        i, j = random.sample(Valid, 2)
        self.state[i], self.state[j] = (
            self.state[j],
            self.state[i]
        )
    #calculates the enery required by a solution, the lowest the better
    def energy(self):
        return calculateSeasonDistancePenalties(self.tracks, self.state, self.home, 15, 35)

# class that will hold the weekends for genetic algorithms
class CalendarGA:
    def __init__(self, tracks, home):
        self.tracks = tracks
        self.home = home
        #computes the fitness of the case based on the distance and penalties
    def fitness(self, individual):
        return calculateSeasonDistancePenalties(self.tracks, individual, self.home,15,35)


# function that will calculate the total distance for the season assuming a given racetrack as the home racetrack
# the following will be assumed:
# - on a weekend where there is no race the team will return home
# - on a weekend in a double or triple header a team will travel straight to the next race and won't go back home
# - the preseason test will always take place in Bahrain
# - for the summer shutdown and off season the team will return home
def calculateSeasonDistance(tracks, weekends, home):
    TOTdistance = 0
    ACTlocation = home
    Calendar = {}

    for i in range(len(weekends)):
        Calendar[weekends[i]] = i

    for week in range(54):
        if week in Calendar:
            NEXTlocation = Calendar[week]
            TOTdistance += haversine(tracks, ACTlocation, NEXTlocation)
            ACTlocation = NEXTlocation

        if (week + 1) not in Calendar:
            TOTdistance += haversine(tracks, ACTlocation, home)
            ACTlocation = home
    return TOTdistance


# # function that will calculate the season distance and will include the cost of penalties in the calculation
def calculateSeasonDistancePenalties(tracks, weekends, home, min, max):
    Dist = calculateSeasonDistance(tracks, weekends, home)

    # Add 100,000 Km penalty if temperature constraint is not respected
    if not checkTemperatureConstraint(tracks, weekends, min, max):
        Dist += 100000

    # Add 100,000 Km penalty if summer shutdown is not respected
    if not checkSummerShutdown(weekends):
        Dist += 100000

    return Dist


# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of min degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, min, max):
    for i in range(len(weekends)):
        Week = weekends[i]
        track = i
        Temp = Week + 3
        TempLoc = tracks[Temp][track]

        if TempLoc < min:
            return False
        if TempLoc > max:
            return False

    return True


# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races. this will be weeks 29, 30, and 31, they are not
# permitted to have a race during these weekends
def checkSummerShutdown(weekends):
    if 29 in weekends:
        return False
    if 30 in weekends:
        return False
    if 31 in weekends:
        return False
    return True


# will go through the genetic code of this child and will make sure that all the required weekends are in it.
# it's highly likely that with crossover that there will be weekends missing and others duplicated. we will
# randomly replace the duplicated ones with the missing ones
def childGeneticCodeFix(child):
    #create the list of all the required weekends, one with the one that are in the calendar and the one that are duplicates
    ListW = set(readRaceWeekends())
    AlreadyIn = set()
    duplicates = []

    #checks if the weekend is already in the calendar
    for i, w in enumerate(child):
        if w in AlreadyIn:
            duplicates.append(i)
        else:
            AlreadyIn.add(w)

    #determines the missing weekends
    missing = list(ListW - AlreadyIn)
    random.shuffle(missing)

    ##swap the duplicates with missing rendom weekends
    for idx in duplicates:
        child[idx] = missing.pop()


# function that will take in the given set of rows and will convert the given row index into floating point values
# this assumes the header in the CSV file is still present so it will skip the first column
def convertRowToFloat(rows, row_index):
    row = rows[row_index]
    for i in range(1, len(row)):
        try:
            row[i] = float(row[i])
        except:
            continue


# function that will count how many elements in the given array are greater equal a specific value
def countGreaterEqual(array, value):
    return sum(1 for x in array if x >= value)


# function that will perform roulette wheel crossover to generate children
def crossoverStrategy(ind1, ind2):
    size = len(ind1)

    #1/2 chance to make a random swap
    for i in range(size):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    #repairs the children in case of duplicates
    childGeneticCodeFix(ind1)
    childGeneticCodeFix(ind2)

    return ind1, ind2


# function that will evaluate a calendar (individual) for the genetic algorithm
def evaluateStrategy(individual):
    tracks = readTrackLocations()
    return (calculateSeasonDistancePenalties(tracks, individual, 8, 15, 35),)

# function that will generate the initial itineraries for particle swarm optimisation
# this will take the initial solution and will shuffle it to create new solutions
def generateInitialItineraries(NmbPrt, initial_solution):

    itineraries = []
    for _ in range(NmbPrt):
        itineraries.append(generateShuffledItinerary(initial_solution))
    return itineraries


# function that will generate a shuffled itinerary. However, this will make sure that the fixed races
# will retain their fixed weeks in the calendar
def generateShuffledItinerary(weekends):
    #creates a double of the current calendar and fixes Valencia
    CalDouble = weekends.copy()
    DoNotMove = {21 }
    #gets the indexes of the moving races, and shuffles them to return a new calendar
    idx = [i for i in range(len(CalDouble)) if i not in DoNotMove]

    values = [CalDouble[i] for i in idx]
    random.shuffle(values)

    for i, v in zip(idx, values):
        CalDouble[i] = v

    return CalDouble


# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.
def haversine(rows, location1, location2):
    Lat1 = rows[2][location1]
    Lat2 = rows[2][location2]

    Lon1 = rows[3][location1]
    Lon2 = rows[3][location2]

    Lt1 = math.radians(Lat1)
    Lt2 = math.radians(Lat2)
    Lg1 = math.radians(Lon1)
    Lg2 = math.radians(Lon2)

    DiffLat = Lt2 - Lt1
    DiffLon = Lg2 - Lg1

    res = math.sin(DiffLat / 2) ** 2 + math.cos(Lt1) * math.cos(Lt2) * math.sin(DiffLon / 2) ** 2
    res2 = 2 * math.asin(math.sqrt(res)) * 6371

    return res2


# function that will initialise an individual (a calendar) for the genetic algorithm
def initIndividual(ind_class):
    weekends = readRaceWeekends()
    return ind_class(generateShuffledItinerary(weekends))


# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexHighestTemp(tracks, weekends, max):
    HighTemp = -1
    HighTempId = -1

    for i in range(len(weekends)):
        if i == 21:
            continue
        week = weekends[i]
        track = i
        Temp = week + 3
        TempLoc = tracks[Temp][track]

        if TempLoc > max and TempLoc > HighTemp:
            HighTemp = TempLoc
            HighTempId = i

    return HighTempId


# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexLowestTemp(tracks, weekends, min):
    LowTemp = 1000
    LowTempId = -1

    for i in range(len(weekends)):
        if i == 21:
            continue
        week = weekends[i]
        track = i
        Temp = week + 3
        TempLoc = tracks[Temp][track]

        if TempLoc < min and TempLoc < LowTemp:
            LowTemp = TempLoc
            LowTempId = i

    return LowTempId

# function that will mutate an individual
def mutateIndividual(individual, indpb):
    tracks = readTrackLocations()
    #find a race with low temperature and swaps it
    MinIndex = indexLowestTemp(tracks, individual, 15)
    if MinIndex != -1:
        Valid = [i for i in range(len(individual)) if i != 21 and i != MinIndex]
        SwapChoice = random.choice(Valid)
        individual[MinIndex], individual[SwapChoice] = individual[SwapChoice], individual[MinIndex]
        return (individual,)
    # find a race with high temperature and swaps it
    MaxIndex = indexHighestTemp(tracks, individual, 35)
    if MaxIndex != -1:
        Valid = [i for i in range(len(individual)) if i != 21 and i != MaxIndex]
        SwapChoice = random.choice(Valid)
        individual[MaxIndex], individual[SwapChoice] = individual[SwapChoice], individual[MaxIndex]
        return (individual,)
    # do a classic random swap
    Valid = [i for i in range(len(individual)) if i != 21]
    i, j = random.sample(Valid, 2)
    individual[i], individual[j] = individual[j], individual[i]

    return (individual,)

# objective function for particle swarm optimisation
def objectiveCalendar(particles):

    global itineraries
    tracks = readTrackLocations()
    costs = []

    #explores the particle swarm, selects an itinerary, apply swaps and calculates the cost of the itinerary
    for particle in particles:
        itinerary_index = int(particle[0]) % len(itineraries)
        itinerary = itineraries[itinerary_index].copy()

        swapElements(itinerary, particle[1:])

        cost = calculateSeasonDistancePenalties(tracks, itinerary, home, 15, 35)
        costs.append(cost)

    return np.array(costs)


# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home):
    print("\nFinal Season Schedule : ")

    calendar = {week: i for i, week in enumerate(weekends)}
    print(f"{'Week':<6} {'Race #':<8} {'Track':<25} {'Temperature (°C)':<15}")
    print("_" * 58)

    for week in sorted(calendar.keys()):
        TrackId = calendar[week]
        Track = tracks[0][TrackId]
        TempLoc = week + 3
        Temp = tracks[TempLoc][TrackId]

        print(f"{week+1:<6} {TrackId + 1:<8} {Track:<25} {Temp:<15.2f}")

    print("_" * 58)
    print("\nTotal Season Distance: " + str(round(calculateSeasonDistance(tracks, weekends, home), 2)) + " Km\n")
    print("Total Distance with Penalties: " + str(round(calculateSeasonDistancePenalties(tracks, weekends, home, 15, 35), 2)) + " Km\n")


# function that will take in the given CSV file and will read in its entire contents
# and return a list of lists
def readCSVFile(file):
    # the rows to return
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open(file)
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows.
    for row in csv_reader:
        rows.append(row)

    # close the file when reading is finished
    csv_file.close()

    # return the rows at the end of the function
    return rows


# function that will read in the race weekends file and will perform all necessary conversions on it
def readRaceWeekends():
    rows = readCSVFile(RACE_WEEKENDS_FILE)
    finalrows = []

    for row in rows[1:]:
        finalrows.append(int(row[1]))

    return finalrows


# function that will read the track locations file and will perform all necessary conversions on it.
# this should also strip out the first column on the left which is the header information for all the rows
def readTrackLocations():
    rows = readCSVFile(TRACK_LOCATIONS_FILE)
    convertRowToFloat(rows, 2)
    convertRowToFloat(rows, 3)
    for i in range(4, len(rows)):
        convertRowToFloat(rows, i)

    finalrows = []

    for row in rows:
        finalrows.append(row[1:])

    return finalrows


# function that performs a roulette wheel randomisation on the two given values and returns the chosen on
def rouletteWheel(a, b):
    """
    Perform roulette wheel randomisation on two given values and return the chosen one.
    """
    if random.random() < 0.5:
        return a
    else:
        return b

# TODO
# function that will take an itinearary and will swap the elements based on the values in the particle. If only one element is selected
# for swapping it will be randomly swapped with another element. if two or more are selected we will take those elements and shuffle them
def swapElements(itinerary, particle):

    Id = swapIndexes(particle)

    if len(Id) == 0:
        return
    elif len(Id) == 1:
        swapIndex(itinerary, Id[0])
    else:
        VId = [i for i in Id if i != 21]
        if len(VId) > 1:
            values = [itinerary[i] for i in VId]
            random.shuffle(values)
            for i, v in zip(VId, values):
                itinerary[i] = v

# function that will return a list of the indexes to be swaped according to the particle
def swapIndexes(particle):
    return [i for i, v in enumerate(particle) if v > 0.5]

# function that will take an itineary and will swap a pair of weekends without changing valencia
def swapPair(itinerary):
    valid = [i for i in range(len(itinerary)) if i != 21]
    i, j = random.sample(valid, 2)
    itinerary[i], itinerary[j] = itinerary[j], itinerary[i]

# function that will swap the values at the given index with another randomly chosen index
def swapIndex(itinerary, index):
    
    if index == 21:
        return

    valid = [i for i in range(len(itinerary)) if i != 21 and i != index]
    if valid:
        Swap = random.choice(valid)
        itinerary[index], itinerary[Swap] = itinerary[Swap], itinerary[index]


# function that will run simulated annealing to shorten the season distance (Mugello as home race)
def SAcases():
    tracks = readTrackLocations()
    weekends = readRaceWeekends()

    print("Simulated Annealing")
    print("Home Race :  Mugello")

    annealer = CalendarAnnealer(weekends.copy(), 8, tracks)

    annealer.steps = 100000
    annealer.copy_strategy = "deepcopy"
    best_state, best_energy = annealer.anneal()

    print("\nBest distance found: " + str(round(best_energy, 2)) + " Km")
    print("Temperature constraint satisfied: " + str(checkTemperatureConstraint(tracks, best_state, 15, 35)))
    print("Summer shutdown satisfied: " + str(checkSummerShutdown(best_state)))
    print("\nBest itinerary:")
    printItinerary(tracks, best_state, 8)


# function that will run the genetic algorithm to shorten the season distance
# TODO
def GAcases():

    tracks = readTrackLocations()
    weekends = readRaceWeekends()

    # clears from previous simulations
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: generateShuffledItinerary(weekends))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossoverStrategy)
    toolbox.register("mutate", mutateIndividual, indpb=0.25)
    toolbox.register("select", tools.selRoulette)

    calendarGA = CalendarGA(tracks, 8)

    toolbox.register(
        "evaluate",
        lambda ind: (calendarGA.fitness(ind),)
    )

    population = toolbox.population(n=300)


    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=CXPB,
        mutpb=0.25,
        ngen=1000,
        halloffame=hof,
        verbose=False
    )

    # Get the best individual
    best = hof[0]

    best_cost = best.fitness.values[0]
    best_solution = list(best)

    # does 2000 iterations
    for iteration in range(2000):
        temp_solution = best_solution.copy()

        # Use the same mutation strategy as SA for consistency
        MinIndex = indexLowestTemp(tracks, temp_solution, 15)
        if MinIndex != -1:
            Valid = [i for i in range(len(temp_solution)) if i != 21 and i != MinIndex]
            SwapChoice = random.choice(Valid)
            temp_solution[MinIndex], temp_solution[SwapChoice] = temp_solution[SwapChoice], temp_solution[MinIndex]
        else:
            MaxIndex = indexHighestTemp(tracks, temp_solution, 35)
            if MaxIndex != -1:
                Valid = [i for i in range(len(temp_solution)) if i != 21 and i != MaxIndex]
                SwapChoice = random.choice(Valid)
                temp_solution[MaxIndex], temp_solution[SwapChoice] = temp_solution[SwapChoice], temp_solution[MaxIndex]
            else:
                # Random swap when constraints are satisfied
                Valid = [i for i in range(len(temp_solution)) if i != 21]
                i, j = random.sample(Valid, 2)
                temp_solution[i], temp_solution[j] = temp_solution[j], temp_solution[i]

        temp_cost = calculateSeasonDistancePenalties(tracks, temp_solution, 8, 15, 35)

        if temp_cost < best_cost:
            best_cost = temp_cost
            best_solution = temp_solution.copy()

    print("\nGenetic Algorithm Result")
    print("Home Race :  Mugello")
    print("Best distance found: " + str(round(best_cost, 2)) + " Km")
    print("Temperature constraint satisfied: " + str(checkTemperatureConstraint(tracks, best_solution, 15, 35)))
    print("Summer shutdown satisfied: " + str(checkSummerShutdown(best_solution)))
    print("\nBest itinerary:")
    printItinerary(tracks, best_solution, 8)

# function that will run particle swarm optimisation in an attempt to find a solution
def PSOcases():
    
    global itineraries, home

    tracks = readTrackLocations()
    weekends = readRaceWeekends()
    home = 8  # Mugello

    print("\nParticle Swarm Optimisation")
    print("Home Race :  Mugello")

    # creates initial itineraries for the swarm
    NmbPrt = 100
    itineraries = generateInitialItineraries(NmbPrt, weekends)

    # Initialize the parameters of the simulation
    dimensions = len(weekends) + 1
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    # Initialize the dimension of the simulation
    Lbounds = [0] + [0] * len(weekends)
    UBounds = [NmbPrt] + [1] * len(weekends)
    Bound = (np.array(Lbounds), np.array(UBounds))

    
    optimizer = ps.single.GlobalBestPSO(
        n_particles=NmbPrt,
        dimensions=dimensions,
        options=options,
        bounds=Bound
    )

    # Lauch the simulation and extracts the best reults
    best_cost, best_pos = optimizer.optimize(objectiveCalendar, iters=1000)

    # Extract best itinerary
    best_itinerary_index = int(best_pos[0]) % len(itineraries)
    best_itinerary = itineraries[best_itinerary_index].copy()
    swapElements(best_itinerary, best_pos[1:])

    best_cost = calculateSeasonDistancePenalties(tracks, best_itinerary, home, 15, 35)

    for iteration in range(1000):
        temp_itinerary = best_itinerary.copy()


        tracks_temp = readTrackLocations()
        MinIndex = indexLowestTemp(tracks_temp, temp_itinerary, 15)
        if MinIndex != -1:
            Valid = [i for i in range(len(temp_itinerary)) if i != 21 and i != MinIndex]
            SwapChoice = random.choice(Valid)
            temp_itinerary[MinIndex], temp_itinerary[SwapChoice] = temp_itinerary[SwapChoice], temp_itinerary[MinIndex]
        else:
            MaxIndex = indexHighestTemp(tracks_temp, temp_itinerary, 35)
            if MaxIndex != -1:
                Valid = [i for i in range(len(temp_itinerary)) if i != 21 and i != MaxIndex]
                SwapChoice = random.choice(Valid)
                temp_itinerary[MaxIndex], temp_itinerary[SwapChoice] = temp_itinerary[SwapChoice], temp_itinerary[MaxIndex]
            else:
                idx1 = random.randint(0, len(temp_itinerary) - 1)
                if idx1 != 21:
                    Valid = [i for i in range(len(temp_itinerary)) if i != 21 and i != idx1]
                    idx2 = random.choice(Valid)
                    temp_itinerary[idx1], temp_itinerary[idx2] = temp_itinerary[idx2], temp_itinerary[idx1]

        temp_cost = calculateSeasonDistancePenalties(tracks, temp_itinerary, home, 15, 35)

        if temp_cost < best_cost:
            best_cost = temp_cost
            best_itinerary = temp_itinerary.copy()

    final_cost = calculateSeasonDistancePenalties(tracks, best_itinerary, home, 15, 35)

    print("\nBest distance found: " + str(round(final_cost, 2)) + " Km")
    print("Temperature constraint satisfied: " + str(checkTemperatureConstraint(tracks, best_itinerary, 15, 35)))
    print("Summer shutdown satisfied: " + str(checkSummerShutdown(best_itinerary)))
    print("\nBest itinerary:")
    printItinerary(tracks, best_itinerary, home)


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Optimise the MotoGP calendar to minimise travel distance.")
    parser.add_argument(
        "algorithm",
        nargs="?",
        default="sa",
        choices=["sa", "ga", "pso", "all", "test"],
        help="sa = simulated annealing (default), ga = genetic algorithm, "
             "pso = particle swarm optimisation, all = run the three, test = run the unit tests",
    )
    args = parser.parse_args()

    if args.algorithm == "test":
        unittest.main(argv=[sys.argv[0]])
    if args.algorithm in ("sa", "all"):
        SAcases()
    if args.algorithm in ("ga", "all"):
        GAcases()
    if args.algorithm in ("pso", "all"):
        PSOcases()
