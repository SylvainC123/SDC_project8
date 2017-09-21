/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define n_part 200     /// number of particules
#define min_val 0.00001  ///threshold for division by zero
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = n_part;
  default_random_engine gen;
  /// define normal distributions for x,y,teta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
	
  for(int i = 0; i<num_particles; i++){
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    weights.push_back(1.0);
    particles.push_back(p);
  }
  is_initialized = true;
  cout<<"initialisation done"<<endl;
  return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_t(0, std_pos[2]);
  
  /// remove division by zero case
  if (fabs(yaw_rate) < min_val) { 
		yaw_rate = min_val;
  }

  for (int i = 0; i < num_particles; i++) {
    /// Add measurements to each particle and add random Gaussian noise
    particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + noise_x(gen);
    particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t))+ noise_y(gen);
    particles[i].theta += yaw_rate * delta_t + noise_t(gen);   
  }
  return;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	/// not used integrated into the update weights
	return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  weights.clear();
  for(int i = 0; i<num_particles; i++){
    std::vector<LandmarkObs> predicted;
  
    double px = particles[i].x;
    double py = particles[i].y;
    double pt = particles[i].theta;
    
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();
    double weight = 1;
    
    for(unsigned int j = 0; j<observations.size(); j++){
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;
      double obs_x_map = px + obs_x * cos(pt) - obs_y * sin(pt);
      double obs_y_map = py + obs_x * sin(pt) + obs_y * cos(pt);
      
      if(pow(pow(obs_x_map-px,2)+pow(obs_y_map-py,2),0.5) > sensor_range) continue;
      particles[i].sense_x.push_back(obs_x_map);
      particles[i].sense_y.push_back(obs_y_map);
      
      double min = 1e10;
      int min_index=-1;
      
      for(unsigned int k = 0; k<map_landmarks.landmark_list.size(); k++){
        double lm_x = map_landmarks.landmark_list[k].x_f;
        double lm_y = map_landmarks.landmark_list[k].y_f;       
        double range = pow(pow(lm_x - obs_x_map,2)+pow(lm_y - obs_y_map,2),0.5);
        if(range < min){
          min = range;
          min_index = k;
        }
      }
      double lm_x = map_landmarks.landmark_list[min_index].x_f;
      double lm_y = map_landmarks.landmark_list[min_index].y_f;

      particles[i].associations.push_back(map_landmarks.landmark_list[min_index].id_i);
      weight = weight * exp(-0.5 * (pow((lm_x - obs_x_map) / std_landmark[0],2) + pow((lm_y - obs_y_map) / std_landmark[1],2))) / (2*M_PI*std_landmark[0]*std_landmark[1]);

    } 
    particles[i].weight=weight;
    weights.push_back(weight); 
  }
  return;		
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	vector<Particle> new_particles;
	discrete_distribution<int> dist(weights.begin(), weights.end());

    /// resample particles
    for (int i = 0; i < num_particles; i++){
        new_particles.push_back(particles[dist(gen)]);
    }
    particles = new_particles;
    return;
    
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
