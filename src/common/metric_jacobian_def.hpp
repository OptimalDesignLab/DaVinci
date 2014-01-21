/**
 * \file metric_jacobian_def.hpp
 * \brief definition of MetricJacobian evaluator methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/assert.hpp>
#include "Teuchos_Tuple.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

namespace davinci {
//==============================================================================
template <typename NodeT, typename ScalarT>
MetricJacobian<NodeT,ScalarT>::MetricJacobian() {
#if 0
  output_names_.resize(3);
  output_names_[0] = string("jacob");
  output_names_[1] = string("jacob_inv");
  output_names_[2] = string("jacob_det");
  input_names_.resize(1);
  input_names_[0] = string("nodes");
#endif
}
//==============================================================================
template <typename NodeT, typename ScalarT>
void MetricJacobian<NodeT,ScalarT>::MemoryRequired(
    int& mesh_offset, map<string,int>& mesh_map_offset,
    int& soln_offset, map<string,int>& soln_map_offset,
    int& resid_offset, map<string,int>& resid_map_offset) const {
  BOOST_ASSERT_MSG(mesh_offset >= 0, "mesh_offset must be >= 0");
  BOOST_ASSERT_MSG(soln_offset >= 0, "soln_offset must be >= 0");
  BOOST_ASSERT_MSG(resid_offset >= 0, "resid_offset must be >= 0");
  mesh_map_offset["jacob"] = mesh_offset;
  mesh_offset += num_elems_*num_cub_points_*dim_*dim_;
  mesh_map_offset["jacob_inv"] = mesh_offset;
  mesh_offset += num_elems_*num_cub_points_*dim_*dim_;
  mesh_map_offset["jacob_det"] = mesh_offset;
  mesh_offset += num_elems_*num_cub_points_;
}
//==============================================================================
template <typename NodeT, typename ScalarT>
void MetricJacobian<NodeT,ScalarT>::SetDataViews(
    ArrayRCP<NodeT>& mesh_data, map<string,int>& mesh_map_offset,
    ArrayRCP<ScalarT>& soln_data, map<string,int>& soln_map_offset,
    ArrayRCP<MetricJacobian<NodeT,ScalarT>::ResidT>& resid_data,
    map<string,int>& resid_map_offset) {
  using Teuchos::tuple;
  // views of inputs
  node_coords_ = GenerateConstView(mesh_data, mesh_map_offset.at("node_coords"),
                                   tuple(num_elems_, num_nodes_per_elem_, dim_));
  // views of outputs
  jacob_ = GenerateView(mesh_data, mesh_map_offset.at("jacob"),
                        tuple(num_elems_, num_cub_points_, dim_, dim_));
  jacob_inv_ = GenerateView(mesh_data, mesh_map_offset.at("jacob_inv"),
                            tuple(num_elems_, num_cub_points_, dim_, dim_));
  jacob_det_ = GenerateView(mesh_data, mesh_map_offset.at("jacob_det"),
                            tuple(num_elems_, num_cub_points_));
}
//==============================================================================
template <typename NodeT, typename ScalarT>
void MetricJacobian<NodeT,ScalarT>::Evaluate(
    const RCP<CellTopology>& topology,
    const FieldContainer<double>& cub_points,
    const FieldContainer<double>& cub_weights,
    const FieldContainer<double>& basis_vals,
    const FieldContainer<double>& basis_grads) {
  typedef Intrepid::CellTools<NodeT> CellTools;
  CellTools::setJacobian(*jacob_, cub_points, *node_coords_, *topology);
  CellTools::setJacobianInv(*jacob_inv_, *jacob_);
  CellTools::setJacobianDet(*jacob_det_, *jacob_);
}
//==============================================================================
}
