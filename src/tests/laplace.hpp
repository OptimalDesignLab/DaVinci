/**
 * \file laplace.hpp
 * \brief class declaration for Laplace/BC evaluator and Factory
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_LAPLACE_HPP
#define DAVINCI_SRC_COMMON_LAPLACE_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "evaluator.hpp"
#include "work_set_factory.hpp"

namespace davinci {

using std::ostream;
using Teuchos::RCP;
using Teuchos::Array;
using Teuchos::ArrayRCP;
using Intrepid::FieldContainer;

/*!
 * \class Laplace
 * \brief computes the Laplace bilinear form residual for a given workset
 * \tparam NodeT - the scalar type for node-based data (double, sacado AD, etc)
 * \tparam ScalarT - the scalar type for sol-based data (double, sacado AD, etc)
 */
template <typename NodeT, typename ScalarT>
class Laplace : public Evaluator<NodeT,ScalarT> {
 public:

  /*!
   * \typedef ResidT
   * \brief type used for fields dependent on both NodeT and ScalarT
   */
  typedef typename Evaluator<NodeT,ScalarT>::ResidT ResidT;
  
  /*!
   * \brief default constructor that defines the output and input dependencies
   */
  Laplace();

  /*!
   * \brief memory requirements and offsets for dependent fields
   * \param[in,out] mesh_offset - offset location of unused mesh-type data
   * \param[in,out] mesh_map_offset - map between mesh field and its offset
   * \param[in,out] soln_offset - offset location of unused solution data
   * \param[in,out] soln_map_offset - map between solution field and its offset
   * \param[in,out] resid_offset - offset location of unused residual data
   * \param[in,out] resid_map_offset - map between residual field and its offset
   *
   * \pre the offset integers should indicate the last unused location in the
   * corresponding data array, i.e. either the mesh arrary or solution arrary
   * \post the offset integers are incremented according to the memory
   * requirements of the evaluator
   */
  virtual void MemoryRequired(
      int& mesh_offset, map<string,int>& mesh_map_offset,
      int& soln_offset, map<string,int>& soln_map_offset,
      int& resid_offset, map<string,int>& resid_map_offset) const;

  /*!
   * \brief shallow copies 1-d array data into FieldContainers for easier access
   * \param[in] mesh_data - 1-d array of mesh-based data
   * \param[in] mesh_map_offset - map between mesh field and its offset
   * \param[in] soln_data - 1-d arrary of solution-based data
   * \param[in] soln_map_offset - map between solution field and its offset
   * \param[in] resid_data - 1-d arrary of residual-type data
   * \param[in] resid_map_offset - map between residual field and its offset
   */
  virtual void SetDataViews(
      ArrayRCP<NodeT>& mesh_data, map<string,int>& mesh_map_offset,
      ArrayRCP<ScalarT>& soln_data, map<string,int>& soln_map_offset,
      ArrayRCP<ResidT>& resid_data, map<string,int>& resid_map_offset);

  /*!
   * \brief compute the dependent variables based on the independent ones
   * \param[in] topology - shards CellTopology of the elements in the workset
   * \param[in] cub_points - cubature points on reference element
   * \param[in] cub_weights - cubature weights on reference element
   * \param[in] basis_vals - the ref element basis values at the cubature points
   * \param[in] basis_grads - the basis gradient at the cubature points
   */  
  void Evaluate(const CellTopology& topology,
                const FieldContainer<double>& cub_points,
                const FieldContainer<double>& cub_weights,
                const FieldContainer<double>& basis_vals,
                const FieldContainer<double>& basis_grads);

  /*!
   * \brief compute the dependent variables based on the independent ones
   */  
  void Evaluate();
  
 protected:
  // these using statements are needed because the data members of the templated
  // base class are hidden
  using Evaluator<NodeT,ScalarT>::num_elems_;
  using Evaluator<NodeT,ScalarT>::num_nodes_per_elem_;
  using Evaluator<NodeT,ScalarT>::num_cub_points_;
  using Evaluator<NodeT,ScalarT>::num_ref_basis_;
  using Evaluator<NodeT,ScalarT>::dim_;
  using Evaluator<NodeT,ScalarT>::basis_grads_;
  using Evaluator<NodeT,ScalarT>::cub_weights_;
  
  // inputs
  RCP<FieldContainer<const NodeT> > jacob_inv_; ///< metric Jacobian inverse
  RCP<FieldContainer<const NodeT> > jacob_det_; ///< determinant of Jacobian
  RCP<FieldContainer<const ScalarT> > solution_coeff_; ///< solution coefficients

  // outputs
  RCP<FieldContainer<NodeT> > jacob_cub_; ///< Jacobian det. weighted by cub
  RCP<FieldContainer<NodeT> > grads_transformed_; ///< basis grads in phys. space
  RCP<FieldContainer<NodeT> > grads_transformed_weighted_; ///< weighed by cub.
  RCP<FieldContainer<ResidT> > solution_grad_; ///< gradient of solution at cub.
  RCP<FieldContainer<ResidT> > residual_; ///< residual over workset
};

/*!
 * \class LaplaceBC
 * \brief computes Dirichlet boundary conditions for a given workset
 * \tparam NodeT - the scalar type for node-based data (double, sacado AD, etc)
 * \tparam ScalarT - the scalar type for sol-based data (double, sacado AD, etc)
 */
template <typename NodeT, typename ScalarT>
class LaplaceBC : public Evaluator<NodeT,ScalarT> {
 public:

  /*!
   * \typedef ResidT
   * \brief type used for fields dependent on both NodeT and ScalarT
   */
  typedef typename Evaluator<NodeT,ScalarT>::ResidT ResidT;
  
  /*!
   * \brief default constructor that defines the output and input dependencies
   */
  LaplaceBC();

  /*!
   * \brief memory requirements and offsets for dependent fields
   * \param[in,out] mesh_offset - offset location of unused mesh-type data
   * \param[in,out] mesh_map_offset - map between mesh field and its offset
   * \param[in,out] soln_offset - offset location of unused solution data
   * \param[in,out] soln_map_offset - map between solution field and its offset
   * \param[in,out] resid_offset - offset location of unused residual data
   * \param[in,out] resid_map_offset - map between residual field and its offset
   *
   * \pre the offset integers should indicate the last unused location in the
   * corresponding data array, i.e. either the mesh arrary or solution arrary
   * \post the offset integers are incremented according to the memory
   * requirements of the evaluator
   */
  virtual void MemoryRequired(
      int& mesh_offset, map<string,int>& mesh_map_offset,
      int& soln_offset, map<string,int>& soln_map_offset,
      int& resid_offset, map<string,int>& resid_map_offset) const;

  /*!
   * \brief shallow copies 1-d array data into FieldContainers for easier access
   * \param[in] mesh_data - 1-d array of mesh-based data
   * \param[in] mesh_map_offset - map between mesh field and its offset
   * \param[in] soln_data - 1-d arrary of solution-based data
   * \param[in] soln_map_offset - map between solution field and its offset
   * \param[in] resid_data - 1-d arrary of residual-type data
   * \param[in] resid_map_offset - map between residual field and its offset
   */
  virtual void SetDataViews(
      ArrayRCP<NodeT>& mesh_data, map<string,int>& mesh_map_offset,
      ArrayRCP<ScalarT>& soln_data, map<string,int>& soln_map_offset,
      ArrayRCP<ResidT>& resid_data, map<string,int>& resid_map_offset);

  /*!
   * \brief compute the dependent variables based on the independent ones
   * \param[in] topology - shards CellTopology of the elements in the workset
   * \param[in] cub_points - cubature points on reference element
   * \param[in] cub_weights - cubature weights on reference element
   * \param[in] basis_vals - the ref element basis values at the cubature points
   * \param[in] basis_grads - the basis gradient at the cubature points
   */  
  void Evaluate(const CellTopology& topology,
                const FieldContainer<double>& cub_points,
                const FieldContainer<double>& cub_weights,
                const FieldContainer<double>& basis_vals,
                const FieldContainer<double>& basis_grads);

  /*!
   * \brief compute the dependent variables based on the independent ones
   */  
  void Evaluate();
  
 protected:
  // these using statements are needed because the data members of the templated
  // base class are hidden
  using Evaluator<NodeT,ScalarT>::num_elems_;
  using Evaluator<NodeT,ScalarT>::num_nodes_per_elem_;
  using Evaluator<NodeT,ScalarT>::num_cub_points_;
  using Evaluator<NodeT,ScalarT>::num_ref_basis_;
  using Evaluator<NodeT,ScalarT>::dim_;
  
  // inputs
  RCP<FieldContainer<const ScalarT> > solution_coeff_; ///< solution coefficients

  // outputs
  RCP<FieldContainer<ResidT> > residual_; ///< residual over workset
};

/*!
 * \class LaplaceFactory
 * \brief a WorkSetFactory for the Laplace PDE
 * \tparam MeshT - a generic class of mesh
 */
template <typename MeshT>
class LaplaceFactory : public WorkSetFactoryBase<MeshT> {
 public:
  typedef MeshT MeshType; ///< this typedef is needed by WorkSetFactory

  /*!
   * \brief the number of PDEs/dependent variables
   */
  int NumPDEs() const { return 1; }
  
 protected:
  /*!
   * \brief Generates the list of evaluators needed by the PDE
   * \param[in] p - a list of options needed to create the evaluators
   * \param[out] evaluators - list of evaluators
   */
  template <typename NodeT, typename ScalarT>
  void CreateEvaluators(const ParameterList& p,
                        Array<RCP<Evaluator<NodeT,ScalarT> > >& evaluators
                        ) const;
};

} // namespace davinci

// include the templated defintions
#include "laplace_def.hpp"

#endif
