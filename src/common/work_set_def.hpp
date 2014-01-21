/**
 * \file work_set_def.hpp
 * \brief definition WorkSet methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/assert.hpp>
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "work_set.hpp"

namespace davinci {
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
WorkSet<NodeT,ScalarT,MeshT>::WorkSet(ostream& out) : evaluators_() {
  out_ = Teuchos::rcp(&out, false);
  // initialize sizes to -1 to avoid using uninitialized values accidentally
  dim_ = -1;
  cub_dim_ = -1;
  num_cub_points_ = -1;
  num_ref_basis_ = -1;
  num_elems_ = -1;
  num_sets_ = -1;
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
void WorkSet<NodeT,ScalarT,MeshT>::DefineTopology(
    const RCP<const CellTopologyData>& cell) {
  topology_ = Teuchos::rcp(new CellTopology(cell.get()));
  num_nodes_per_elem_ = topology_->getNodeCount();
  dim_ = topology_->getDimension(); // !!! this is not general enough
  *out_ << "WorkSet dimension = " << dim_ << "\n";
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
void WorkSet<NodeT,ScalarT,MeshT>::DefineCubature(const int& degree) {
  BOOST_ASSERT_MSG(degree > 0 && degree < 10,
                   "cubature degree must be greater than 0 and less than 10");
  // Get numerical integration points and weights for the defined topology
  using Intrepid::DefaultCubatureFactory;
  using Intrepid::Cubature;
  DefaultCubatureFactory<ScalarT> cubFactory;
  Teuchos::RCP<Cubature<ScalarT> >
      cub = cubFactory.create(*topology_, degree);
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
  *out_ << "\n";
#endif
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
void WorkSet<NodeT,ScalarT,MeshT>::DefineBasis(
    const Basis<ScalarT, FieldContainer<ScalarT> >& basis) {
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::DefineBasis: evaluating basis on reference element\n\n";
#endif
  // copy basis
  basis_ = Teuchos::rcpFromRef(basis);
  // Evaluate basis values and gradients at cubature points
  num_ref_basis_ = basis_->getCardinality();
  vals_.resize(num_ref_basis_, num_cub_points_);
  grads_.resize(num_ref_basis_, num_cub_points_, dim_);
  basis_->getValues(vals_, cub_points_, Intrepid::OPERATOR_VALUE);
  basis_->getValues(grads_, cub_points_, Intrepid::OPERATOR_GRAD);
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
void WorkSet<NodeT,ScalarT,MeshT>::DefineEvaluators() {
  // TEMPORARY
  eval_list.resize(1, CopyNodes());
  eval_list.resize(2, MetricJacobian());
  eval_list.resize(3, Laplace());    
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
void WorkSet<NodeT,ScalarT,MeshT>::ResizeSets(const int& total_elems,
                                        const int& num_elems_per_set) {
  BOOST_ASSERT_MSG(total_elems > 0, "total_elems must be > 0");
  BOOST_ASSERT_MSG(num_elems_per_set > 0 && num_elems_per_set <= total_elems,
                   "num_elems_per_set must be > 0 and < total_elems");
  // determine the number of sets and the remainder set size
  num_elems_ = num_elems_per_set;
  std::div_t div_result = std::div(total_elems-1, num_elems_);
  num_sets_ = div_result.quot+1;
  rem_num_elems_ = div_result.rem+1;
  // allocate arrarys
  node_coords_.resize(num_elems_, num_nodes_per_elem_, dim_);
  jacob_.resize(num_elems_, num_cub_points_, dim_, dim_); 
  jacob_inv_.resize(num_elems_, num_cub_points_, dim_, dim_);
  jacob_det_.resize(num_elems_, num_cub_points_);
  weighted_measure_.resize(num_elems_, num_cub_points_);
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
void WorkSet<NodeT,ScalarT,MeshT>::CopyMeshCoords(const MeshT& mesh,
                                            const int& set_idx) {
  BOOST_ASSERT_MSG(set_idx >= 0 && set_idx < num_sets_,
                   "set_idx number must be positive and less than num_sets_");
  // copy the physical cell coordinates
  int set_num_elems = num_elems_;
  if (set_idx == num_sets_-1) set_num_elems = rem_num_elems_;
  for (int ielem = 0; ielem < set_num_elems; ielem++) {
    int k = set_idx*num_elems_ + ielem;
    for (int i = 0; i < num_nodes_per_elem_; i++)
      for (int j = 0; j < dim_; j++) 
        node_coords_(ielem,i,j) = mesh.ElemNodeCoord(k,i,j);
  }
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT>
void WorkSet<NodeT,ScalarT,MeshT>::BuildSystem(const MeshT& mesh) {
  *out_ << "WorkSet::BuildSystem: must be defined in a derived class\n\n";
  throw(-1);
}
//==============================================================================
} // namespace davinci
