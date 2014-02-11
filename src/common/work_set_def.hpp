/**
 * \file work_set_def.hpp
 * \brief definition WorkSet methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/assert.hpp>
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "work_set.hpp"
#include "Sacado_Fad_SFad.hpp"
#include "Sacado_Fad_DFad.hpp"

namespace davinci {
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
WorkSet<NodeT,ScalarT,MeshT,BasisT>::WorkSet(
    ostream& out) : evaluators_() {
  out_ = Teuchos::rcp(&out, false);
  // initialize sizes to -1 to avoid using uninitialized values accidentally
  dim_ = -1;
  cub_dim_ = -1;
  num_cub_points_ = -1;
  num_elems_ = -1;
  num_sets_ = -1;
  // get basis and topology information
  num_ref_basis_ = this->getCardinality();
  num_nodes_per_elem_ = this->getBaseCellTopology().getNodeCount();
  dim_ = this->getBaseCellTopology().getDimension();
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::WorkSet(): constructed WorkSet object for "
        << this->getBaseCellTopology().getName() << "\n";
#endif
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
void WorkSet<NodeT,ScalarT,MeshT,BasisT>::DefineCubature(
    const int& degree) {
  BOOST_ASSERT_MSG(degree > 0 && degree < 10,
                   "cubature degree must be greater than 0 and less than 10");
  // Get numerical integration points and weights for the defined topology
  using Intrepid::DefaultCubatureFactory;
  using Intrepid::Cubature;
  DefaultCubatureFactory<double> cubFactory;
  Teuchos::RCP<Cubature<double> >
      cub = cubFactory.create(this->getBaseCellTopology(), degree);
  cub_dim_ = cub->getDimension();
  num_cub_points_ = cub->getNumPoints();
  cub_points_.resize(num_cub_points_, cub_dim_);
  cub_weights_.resize(num_cub_points_);
  cub->getCubature(cub_points_, cub_weights_);
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::SetCubature():\n";
  for (int i=0; i<num_cub_points_; i++)
    *out_ << "\tcubature point " << i << ": (" << cub_points_(i,0) << ","
          << cub_points_(i,1) << ") : weight = " << cub_weights_(i) << "\n";
#endif
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
void WorkSet<NodeT,ScalarT,MeshT,BasisT>::EvaluateBasis() {
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::DefineBasis(): evaluating basis on reference element\n";
#endif
  vals_.resize(num_ref_basis_, num_cub_points_);
  grads_.resize(num_ref_basis_, num_cub_points_, dim_);
  this->getValues(vals_, cub_points_, Intrepid::OPERATOR_VALUE);
  this->getValues(grads_, cub_points_, Intrepid::OPERATOR_GRAD);
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
void WorkSet<NodeT,ScalarT,MeshT,BasisT>::DefineEvaluators(
    const std::list<Evaluator<NodeT,ScalarT>* >& evaluators) {
  BOOST_ASSERT_MSG(evaluators.size() > 0, "list of evaluators cannot be empty");
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::DefineEvaluators(): setting list of evaluators\n";
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
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
void WorkSet<NodeT,ScalarT,MeshT,BasisT>::ResizeSets(
    const int& num_pdes, const int& total_elems, const int& num_elems_per_set) {
  BOOST_ASSERT_MSG(num_pdes > 0, "num_pdes must be > 0");
  BOOST_ASSERT_MSG(total_elems > 0, "total_elems must be > 0");
  BOOST_ASSERT_MSG(num_elems_per_set > 0 && num_elems_per_set <= total_elems,
                   "num_elems_per_set must be > 0 and < total_elems");
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::ResizeSets(): defining workset and evaluator dimensions\n";
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
  int num_nodes_per_elem = this->getBaseCellTopology().getNodeCount();
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
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
void WorkSet<NodeT,ScalarT,MeshT,BasisT>::CopySolution(
    const int& set_idx, const ArrayRCP<const double>& sol) {
  BOOST_ASSERT_MSG(num_ref_basis_ == num_nodes_per_elem_,
                   "presently, the basis size must equal the number of nodes");
#ifdef DAVINCI_VERBOSE
  //*out_ << "WorkSet::CopySolution: reading solution into workset\n";
#endif  
  // copy the solution into the workset array
  int set_num_elems = num_elems_;
  if (set_idx == num_sets_-1) set_num_elems = rem_num_elems_;
  int num_AD_dep_vars = num_ref_basis_*num_pdes_;
  for (int ielem = 0; ielem < set_num_elems; ++ielem)
    for (int i = 0; i < num_ref_basis_; ++i)
      for (int j = 0; j < num_pdes_; ++j) {
        // The constructor below makes an automatic differentiation type with
        // num_ref_basis_*num_pdes_ dependent variables, initialized to zeros
        // everywhere except along the diagonal (given here by second parameter)
        // where the value is the solution vector value
        soln_data_[(ielem*num_ref_basis_+i)*num_pdes_+j]
            = ResidT(num_AD_dep_vars, i*num_pdes_ + j,
                     sol[num_pdes_*index_data_[i]+j]);
      }
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
void WorkSet<NodeT,ScalarT,MeshT,BasisT>::Assemble(
    const int& set_idx, const ArrayRCP<double>& rhs,
    const RCP<MatrixT>& jacobian) {
  BOOST_ASSERT_MSG(num_ref_basis_ == num_nodes_per_elem_,
                   "presently, the basis size must equal the number of nodes");
  
  // construct a view of residual data and indices for convenience
  RCP<FieldContainer<const ResidT> > residual =
      GenerateConstView(resid_data_, resid_map_offset_.at("residual"),
                        Teuchos::tuple(num_elems_, num_ref_basis_, num_pdes_));
  RCP<FieldContainer<const typename MeshT::LocIdxT> > index =
      GenerateConstView(index_data_, 0,
                        Teuchos::tuple(num_elems_, num_ref_basis_));
  
  // build some arrays for holding the residual and jacobian blocks
  // TODO: to avoid overhead of creating these, may consider sending them in as
  // work arrays
  typedef Teuchos::SerialDenseVector<int,double> VectorBlock;
  typedef Teuchos::SerialDenseMatrix<int,double> MatrixBlock;
  ArrayRCP<VectorBlock> rhs_block(num_ref_basis_, VectorBlock(num_pdes_));
  ArrayRCP<MatrixBlock> jac_block(num_ref_basis_*num_ref_basis_,
                                  MatrixBlock(num_pdes_,num_pdes_));
  
  // assemble into global matrix
  int set_num_elems = num_elems_;
  if (set_idx == num_sets_-1) set_num_elems = rem_num_elems_;
  int num_AD_dep_vars = num_ref_basis_*num_pdes_; // max number of dependent vars
  for (int ielem = 0; ielem < set_num_elems; ++ielem) {
    // build the blocks for the jacobian and rhs using AD
    for (int i = 0; i < num_ref_basis_; ++i)
      for (int j = 0; j < num_pdes_; ++j) {
        rhs_block[i](j) = (*residual)(ielem, i, j).val();
        for (int i2 = 0; i2 < num_ref_basis_; ++i2)
          for (int j2 = 0; j2< num_pdes_; ++j2)
            jac_block[i*num_ref_basis_ + i2](j,j2) =
                (*residual)(ielem, i, j).dx(i2*num_pdes_+j2);
      }
    // insert blocks into jacobian and rhs
    for (int i = 0; i < num_ref_basis_; ++i) {
      for (int j = 0; j < num_pdes_; ++j)
        rhs[(*index)(ielem,i)*num_pdes_+j] = rhs_block[i](j);
      for (int i2 = 0; i2 < num_ref_basis_; ++i2) 
        jacobian->sumIntoLocalBlockEntry((*index)(ielem,i), (*index)(ielem,i2),
                                         jac_block[i*num_ref_basis_ + i2]);
    }
  }
}
//==============================================================================
template <typename NodeT, typename ScalarT, typename MeshT, typename BasisT>
void WorkSet<NodeT,ScalarT,MeshT,BasisT>::BuildSystem(
    const MeshT& mesh, const RCP<const VectorT>& sol,
    const RCP<VectorT>& rhs, const RCP<MatrixT>& jacobian) {
#ifdef DAVINCI_VERBOSE
  *out_ << "WorkSet::BuildSystem(): creating linear system\n\n";
#endif
  // get some views of the Tpetra objects
  ArrayRCP<const double> sol_view = sol->get1dView(); //sol->getData();
  ArrayRCP<double> rhs_view = rhs->get1dViewNonConst();

  // loop over the workset batches
  int set_num_elems = num_elems_;
  typename std::list<Evaluator<NodeT,ScalarT>* >::iterator evali;  
  for (int set = 0; set < num_sets_; set++) {
    // store mesh node coords, dof indices, and solution in the appropriate
    // arrays
    mesh.CopyElemNodeCoords(mesh_data_, set, num_elems_, num_sets_);
    mesh.CopyElemDOFIndices(index_data_, set, num_elems_, num_sets_);
    CopySolution(set, sol_view);
    // evaluate the necessary fields
    for (evali = evaluators_.begin(); evali != evaluators_.end(); ++evali)
      (*evali)->Evaluate(this->getBaseCellTopology(), cub_points_, cub_weights_,
                         vals_, grads_);
    // insert data into matrix and rhs vector
    Assemble(set, rhs_view, jacobian);
  }
}
//==============================================================================
} // namespace davinci
