/*	
 * File: Random.cpp
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010
 * Description: manages pseudo-random numbers
 * License: see the LICENSE.txt file
 *
 */

#include "Random.h"
#include "Timestamp.h"
#include <cstdlib>
#include <mutex>
#include <random>
using namespace std;

// web port: rand()/srand() replaced by std::mt19937, whose output sequence is fixed by the C++ standard.
// rand() differs between C libraries (glibc vs. musl/emscripten), which made the native and WebAssembly
// builds draw different RANSAC samples from the same seed. The mutex keeps it thread-safe like rand().
namespace {
std::mt19937& engine() { static std::mt19937 e(1u); return e; }  // srand() default seed is 1
std::mutex& engineMutex() { static std::mutex m; return m; }
uint32_t next32() { std::lock_guard<std::mutex> lock(engineMutex()); return engine()(); }
void reseed(unsigned s) { std::lock_guard<std::mutex> lock(engineMutex()); engine().seed(s); }
}

bool DUtils::Random::m_already_seeded = false;

void DUtils::Random::SeedRand(){
	Timestamp time;
	time.setToCurrentTime();
	reseed((unsigned)time.getFloatTime());  // web port: was srand(...)
}

void DUtils::Random::SeedRandOnce()
{
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand();
    m_already_seeded = true;
  }
}

void DUtils::Random::SeedRand(int seed)
{
	reseed((unsigned)seed);  // web port: was srand(seed)
}

void DUtils::Random::SeedRandOnce(int seed)
{
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand(seed);
    m_already_seeded = true;
  }
}

int DUtils::Random::RandomInt(int min, int max){
	int d = max - min + 1;
	return int(((double)next32() / 4294967296.0) * d) + min;  // web port: was rand()/(RAND_MAX+1.0)
}

double DUtils::Random::RandomUnit(){
	return (double)next32() / 4294967295.0;  // [0..1], like rand()/RAND_MAX
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer(int min, int max)
{
  if(min <= max)
  {
    m_min = min;
    m_max = max;
  }
  else
  {
    m_min = max;
    m_max = min;
  }

  createValues();
}

// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  *this = rnd;
}

// ---------------------------------------------------------------------------

int DUtils::Random::UnrepeatedRandomizer::get()
{
  if(empty()) createValues();
  
  DUtils::Random::SeedRandOnce();
  
  int k = DUtils::Random::RandomInt(0, m_values.size()-1);
  int ret = m_values[k];
  m_values[k] = m_values.back();
  m_values.pop_back();
  
  return ret;
}

// ---------------------------------------------------------------------------

void DUtils::Random::UnrepeatedRandomizer::createValues()
{
  int n = m_max - m_min + 1;
  
  m_values.resize(n);
  for(int i = 0; i < n; ++i) m_values[i] = m_min + i;
}

// ---------------------------------------------------------------------------

void DUtils::Random::UnrepeatedRandomizer::reset()
{
  if((int)m_values.size() != m_max - m_min + 1) createValues();
}

// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer& 
DUtils::Random::UnrepeatedRandomizer::operator=
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  if(this != &rnd)
  {
    this->m_min = rnd.m_min;
    this->m_max = rnd.m_max;
    this->m_values = rnd.m_values;
  }
  return *this;
}

// ---------------------------------------------------------------------------


