# MotoGP Calendar Optimization

This project aims to optimize the MotoGP 2025 race calendar by reducing the total distance travelled by the teams while respecting several constraints.

The optimization was done using three different algorithms:

* Simulated Annealing
* Particle Swarm Optimization
* Genetic Algorithm

The goal is to compare the three approaches and see which one can find the shortest valid calendar.

## Objective

During a MotoGP season, teams have to travel between every circuit. The order of the races therefore has a big impact on the total distance travelled.

The main objective of this project is to find a race order that minimizes the total travel distance while respecting the following constraints:

* The temperature at each race must be between 15°C and 35°C.
* The summer shutdown must be respected.
* Mugello must remain the home race.
* Valencia must remain the final race of the season.

The distance between circuits is calculated using their geographical coordinates.

## Data

The project uses two main datasets:

* `race-weekends.csv` contains the race weekends, their weeks and temperature information.
* `track-locations.csv` contains the geographical coordinates of the circuits.

The initial calendar is used as a starting point for the optimization.

### Note about Valencia

In the provided dataset, Valencia is set on week 45. I initially expected it to be week 46 because of the way the weeks are represented, but this was not simply an indexing issue because other races would then also be shifted and the summer shutdown constraint would no longer match.

For this reason, the value from the original dataset was kept unchanged during the optimization.

## Constraints

### Temperature

Each race must take place when the average temperature is between:

```text
15°C ≤ Temperature ≤ 35°C
```

This constraint prevents the algorithms from placing races in periods where the conditions would not be suitable.

### Summer shutdown

The MotoGP summer shutdown must also be respected. No race can be scheduled during the specified shutdown period.

### Home race

Mugello is considered the team's home race and must remain in its required position.

### Final race

Valencia must remain the final race of the calendar.

## Optimization Algorithms

### Simulated Annealing

Simulated Annealing was used to optimize the calendar by swapping race weekends and evaluating the resulting travel distance.

At the beginning of the algorithm, worse solutions can still be accepted. This allows the algorithm to explore the search space and avoid getting stuck in a local optimum.

As the algorithm progresses, the probability of accepting worse solutions decreases. The algorithm then becomes more selective and converges towards a stable solution.

### Particle Swarm Optimization

Particle Swarm Optimization (PSO) was also used to reduce the total travel distance.

In this implementation, a swarm of 100 particles represents different possible solutions. Each particle adjusts itself according to its own best result and the best result found by the swarm.

The algorithm was run for 1,000 iterations.

### Genetic Algorithm

The Genetic Algorithm represents each possible calendar as an individual in a population.

The population evolves through several generations using:

* Selection
* Crossover
* Mutation

The best solutions are progressively favored, allowing the algorithm to search for calendars with a shorter total travel distance while respecting the constraints.

## Results

The three algorithms were tested on the same problem and compared using their total travel distance.

### Simulated Annealing

The best solution found by Simulated Annealing has a total travel distance of:

**120,845.46 km**

The solution respects the temperature constraint, the summer shutdown, the home race and the final race constraints.

The algorithm tends to group races that are geographically close to each other, especially when there is no break between race weekends. This reduces the distance travelled between consecutive races.

### Particle Swarm Optimization

Particle Swarm Optimization found a solution with a total travel distance of:

**125,010.43 km**

The solution respects all the constraints.

Some similarities can be seen compared to the Simulated Annealing solution. For example, Spain is placed close to Valencia, while Portugal and Aragón are also relatively close to each other.

PSO was also the fastest algorithm to execute on my computer.

### Genetic Algorithm

The Genetic Algorithm found a solution with a total travel distance of:

**125,681.86 km**

The solution also respects all the constraints.

The algorithm groups several races geographically and takes temperature conditions into account when deciding where races should be placed during the season.

The result is slightly worse than the results obtained with Simulated Annealing and PSO, but it still provides a valid and competitive solution.

Simulated Annealing obtained the best result, with:

**120,845.46 km**

The three algorithms were able to find valid calendars, but Simulated Annealing gave the shortest total travel distance in this experiment.

## Technologies

The project was developed in Python using:

* Python
* NumPy
* Pandas
* Matplotlib
* DEAP
* PySwarms

## Conclusion

This project shows how different optimization algorithms can be applied to a real scheduling and logistics problem.

Simulated Annealing gave the best result in this experiment with a total travel distance of **120,845.46 km**, followed by Particle Swarm Optimization with **125,010.43 km** and the Genetic Algorithm with **125,681.86 km**.

The results also show that the different algorithms can produce different calendar structures while respecting the same constraints.

The project could be improved in the future by adding other factors such as travel costs, flight distances, travel time, carbon emissions or more detailed weather data. This would make the problem closer to a real-world logistics optimization problem.

## Author

Noé Payan

Engineering Student in Data and Artificial Intelligence

GitHub: https://github.com/NoePayan
