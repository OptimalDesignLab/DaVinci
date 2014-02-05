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
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::WorkSet(
    ostream& out) : evaluators_() {
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
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::DefineTopology(
    const RCP<const CellTopologyData>& cell) {
  topology_ = Teuchos::rcp(new CellTopology(cell.get()));
  num_nodes_per_elem_ = topology_->getNodeCount();
  dim_ = topology_->getDimension(); // !!! this is not general enough
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::DefineTopology: dimension = " << dim_ << "\n";
#endif
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::DefineCubature(
    const int& degree) {
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
    *out_ << "\tcubature point " << i << ": (" << cub_points_(i,0) << ","
          << cub_points_(i,1) << ") : weight = " << cub_weights_(i) << "\n";
#endif
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::DefineBasis(
    const Basis<ScalarT, FieldContainer<ScalarT> >& basis) {
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::DefineBasis: evaluating basis on reference element\n";
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
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::DefineEvaluators(
    const std::list<Evaluator<NodeT,ScalarT>* >& evaluators) {
  BOOST_ASSERT_MSG(evaluators.size() > 0, "list of evaluators cannot be empty");
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::DefineEvaluators: setting list of evaluators\n";
#endif
  //    const ArrayRCP<Evaluator<NodeT,ScalarT> >& evaluators) {
  //  typename ArrayRCP<Evaluator<NodeT,ScalarT> >::iterator it;
  evaluators_.clear();
  typename std::list<Evaluator<NodeT,ScalarT>* >::const_iterator evali;
  for (evali = evaluators.begin(); evali != evaluators.end(); ++evali) {
    evaluators_.push_back(*evali);
  }
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::ResizeSets(
    const int& num_pdes, const int& total_elems, const int& num_elems_per_set) {
  BOOST_ASSERT_MSG(num_pdes > 0, "num_pdes must be > 0");
  BOOST_ASSERT_MSG(total_elems > 0, "total_elems must be > 0");
  BOOST_ASSERT_MSG(num_elems_per_set > 0 && num_elems_per_set <= total_elems,
                   "num_elems_per_set must be > 0 and < total_elems");
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::ResizeSets: defining workset and evaluator dimensions\n";
#endif
  num_pdes_ = num_pdes;
  // determine the number of sets and the remainder set size
  num_elems_ = num_elems_per_set;
  std::div_t div_result = std::div(total_elems-1, num_elems_);
  num_sets_ = div_result.quot+1;
  rem_num_elems_ = div_result.rem+1;
  // set the dimensions of the Evaluators
  typename std::list<Evaluator<NodeT,ScalarT>* >::iterator evali;
  for (evali = evaluators_.begin(); evali != evaluators_.end(); ++evali)
    (*evali)->SetDimensions(num_elems_, num_nodes_per_elem_, num_cub_points_,
                            num_ref_basis_, dim_, num_pdes_);
  // determine memory requirements
  int num_nodes_per_elem = topology_->getNodeCount();
  int mesh_memory = num_elems_*num_nodes_per_elem*dim_; // for nodes
  int soln_memory = num_elems_*num_ref_basis_*num_pdes_; // for solution coeffs
  int resid_memory = 0;
  int index_memory = num_elems_*num_ref_basis_*num_pdes_; // for dof indices
  mesh_map_offset_["node_coords"] = 0;
  soln_map_offset_["solution_coeff"] = 0;
  for (evali = evaluators_.begin(); evali != evaluators_.end(); ++evali)
    (*evali)->MemoryRequired(mesh_memory, mesh_map_offset_,
                             soln_memory, soln_map_offset_,
                             resid_memory, resid_map_offset_);
  // allocate memory
  mesh_data_.resize(mesh_memory, 0.0);
  soln_data_.resize(soln_memory, 0.0);
  resid_data_.resize(resid_memory, 0.0);
  index_data_.resize(index_memory, 0.0);
  // define Data Views for Evaluators; is this the best place for this?
  for (evali = evaluators_.begin(); evali != evaluators_.end(); ++evali)
    (*evali)->SetDataViews(mesh_data_, mesh_map_offset_,
                           soln_data_, soln_map_offset_,
                           resid_data_, resid_map_offset_);
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::CopySolution(
    const int& set_idx, const ArrayRCP<const double>& sol) {
  BOOST_ASSERT_MSG(num_ref_basis_ == static_cast<int>(topology_->getNodeCount()),
                     "presently, the basis size must equal the number of nodes");
#ifdef DAVINCI_VERBOSE
  //*out_ << "WorkSet::CopySolution: reading solution into workset\n";
#endif  
  // copy the solution into the workset array
  int set_num_elems = num_elems_;
  if (set_idx == num_sets_-1) set_num_elems = rem_num_elems_;
  for (int i = 0; i < set_num_elems*num_ref_basis_*num_pdes_; ++i)
    soln_data_[i] = sol[index_data_[i]];
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::Assemble(
    const ArrayRCP<double>& rhs, const RCP<MatrixT>& jacobian,
    const MeshT& mesh) {
  BOOST_ASSERT_MSG(num_ref_basis_ == static_cast<int>(topology_->getNodeCount()),
                   "presently, the basis size must equal the number of nodes");
  // assemble into global matrix
#if 0
  if (set == num_sets_-1) set_num_elems = rem_num_elems_;
  for (int ielem = 0; ielem < set_num_elems; ielem++) {
    int k = set_idx*num_elems_ + ielem; // index of element on local process
    std::vector<int> rowIndex(numFieldsG);
    std::vector<int> colIndex(numFieldsG);
    for (int row = 0; row < numFieldsG; row++){
      rowIndex[row] = elemToNode(k,row);
    }
    for (int col = 0; col < numFieldsG; col++){
      colIndex[col] = elemToNode(k,col);
    }
    for (int row = 0; row < numFieldsG; row++){
      timer_jac_insert_g.start();
      StiffMatrixViaAD.SumIntoGlobalValues(1, &rowIndex[row], numFieldsG, &colIndex[0], cellResidualAD(ci,row).dx());
      timer_jac_insert_g.stop();
    }
  }
#endif
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename VectorT,
          typename MatrixT>
void WorkSet<NodeT,ScalarT,MeshT,VectorT,MatrixT>::BuildSystem(
    const MeshT& mesh, const RCP<const VectorT>& sol,
    const RCP<VectorT>& rhs, const RCP<MatrixT>& jacobian) {
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::BuildSystem: creating linear system\n\n";
#endif
  // get some views of the Tpetra objects
  ArrayRCP<const ScalarT> sol_view = sol->get1dView(); //sol->getData();
  ArrayRCP<ScalarT> rhs_view = rhs->get1dViewNonConst();

  // loop over the workset batches
  int set_num_elems = num_elems_;
  typename std::list<Evaluator<NodeT,ScalarT>* >::iterator evali;  
  for (int set = 0; set < num_sets_; set++) {
    // store mesh node coords, dof indices, and solution in the appropriate
    // arrays
    mesh.CopyElemNodeCoords(mesh_data_, set, num_elems_, num_sets_);
    mesh.CopyElemDOFIndices(index_data_, set, num_elems_, num_sets_, num_pdes_);
    CopySolution(set, sol_view);
    // evaluate the necessary fields
    for (evali = evaluators_.begin(); evali != evaluators_.end(); ++evali)
      (*evali)->Evaluate(topology_, cub_points_, cub_weights_, vals_, grads_);
    // insert data into matrix and rhs vector
    Assemble(rhs_view, jacobian, mesh);
  }
}
//==============================================================================

#if 0
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
#endif
//==============================================================================
} // namespace davinci
