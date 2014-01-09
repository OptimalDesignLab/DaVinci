/**
 * \file work_set_def.hpp
 * \brief definition WorkSet methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/assert.hpp>
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
//#include "Intrepid_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "work_set.hpp"

namespace davinci {

//==============================================================================
template <typename ScalarT>
WorkSet<ScalarT>::WorkSet(ostream& out) {
  out_ = Teuchos::rcp(&out, false);
}
//==============================================================================
template <typename ScalarT>
void WorkSet<ScalarT>::DefineTopology(const CellTopologyData* cell) {
  topology_ = Teuchos::rcp(new CellTopology(cell));
}

//==============================================================================
template <typename ScalarT>
void WorkSet<ScalarT>::DefineCubature(const int& degree) {
  BOOST_ASSERT_MSG(degree > 0 && degree < 10,
                   "cubature degree must be greater than 0 and less than 10");
  // Get numerical integration points and weights for the defined topology
  using Intrepid::DefaultCubatureFactory;
  using Intrepid::Cubature;
  DefaultCubatureFactory<ScalarT>  cubFactory;
  Teuchos::RCP<Cubature<ScalarT> >
      cub = cubFactory.create(topology_, degree);
  cub_dim_ = cub->getDimension();
  num_cub_points_ = cub->getNumPoints();
  cub_points_.resize(num_cub_points_, cub_dim_);
  cub_weights_.resize(num_cub_points_);
  cub->getCubature(cub_points_, cub_weights_);
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::SetCubature:\n";
  for (int i=0; i<num_cub_points_; i++)
    *out_ << " cubature point " << i << ": (" << cub_points_(i,0) << ","
          << cub_points_(i,1) << ") : weight = " << cub_weights_(i) << "\n";
#endif
}

//==============================================================================
template <typename ScalarT>
void WorkSet<ScalarT>::ResizeFields(const int& num_elems) {
  *out_ << "Do stuff\n";
}

} // namespace davinci
