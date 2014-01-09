/**
 * \file work_set_test.cpp
 * \brief unit test for the WorkSet<ScalarT> class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "work_set.hpp"

using davinci::WorkSet;

BOOST_AUTO_TEST_SUITE(WorkSet_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  WorkSet<double> MyWorkSet(std::cout);  
}

BOOST_AUTO_TEST_CASE(Topology) {
  WorkSet<double> MyWorkSet(std::cout);
  const CellTopologyData* cell =
      shards::getCellTopologyData<shards::Triangle<3> >();
  MyWorkSet.DefineTopology(cell);
  cell = shards::getCellTopologyData<shards::Tetrahedron<4> >();
  MyWorkSet.DefineTopology(cell);
}

BOOST_AUTO_TEST_SUITE_END()
