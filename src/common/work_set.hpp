/**
 * \file work_set.hpp
 * \brief class declaration for WorkSet
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_WORK_SET_HPP
#define DAVINCI_SRC_COMMON_WORK_SET_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_Basis.hpp"
#include "evaluator.hpp"

namespace davinci {

using std::ostream;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using shards::CellTopology;
using Intrepid::FieldContainer;
using Intrepid::Basis;

/*!
 * \class WorkSet
 * \brief a collection of data and functions for common-type "elements"
 * \tparam NodeT - the scalar type for node-based data (double, sacado AD, etc)
 * \tparam ScalarT - the scalar type for sol-based data (double, sacado AD, etc)
 * \tparam MeshT - a generic class of mesh
 *
 * A workset is used to evaluate residuals and Jacobians for a set of "elements"
 * with identical topology.
 *
 * \remark It is anticipated the these elements may be elements, their faces, or
 * their edges.  For example, we may use a WorkSet for volume integrals in the
 * semi-linear form, and a separate WorkSet for surface integrals.  This has yet
 * to be decided.
 *
 * \remark Should we bother keeping the topology_ and basis_ members, or should
 * the information provided by these be stripped out, similar to the cubature?
 * At least one of these seems redundant.
 */
template <typename NodeT, typename ScalarT, typename MeshT>
class WorkSet {
 public:
  
  /*!
   * \typedef ResidT
   * \brief type used for fields dependent on both NodeT and ScalarT
   */
  typedef typename davinci::Evaluator<NodeT,ScalarT>::ResidT ResidT;

  /*!
   * \typedef VectorT
   * \brief linear algebra vector for system
   */
  typedef typename Tpetra::BlockMultiVector<
    double, typename MeshT::LocIdxT, typename MeshT::GlbIdxT> VectorT;

  /*!
   * \typedef MatrixT
   * \brief linear algebra matrix for system
   */
  typedef typename Tpetra::VbrMatrix<
    double, typename MeshT::LocIdxT, typename MeshT::GlbIdxT> MatrixT;
  
  /*!
   * \brief constructor
   * \param[in] out - a valid output stream
   */
  WorkSet(ostream& out);

  /*!
   * \brief initializes the topology of all elements in this workset
   * \param[in] cell - pointer to shards cell topology attributes
   */
  void DefineTopology(const RCP<const CellTopologyData>& cell);
  
  /*!
   * \brief sets the cubature points and weights based on degree and topology_
   * \param[in] degree - highest polynomial degree cubature is exact
   */
  void DefineCubature(const int& degree);

  /*!
   * \brief evaluates the basis and its derivatives at the cubature points
   * \param[in] basis - the desired basis for the elements
   *
   * \warning This is not yet checked for consistency with topology_
   * \todo Make a check for consistency with topology_!
   */
  void DefineBasis(const Basis<double, FieldContainer<double> >& basis);

  /*!
   * \brief sets the evaluators that define the problem on this workset
   * \param[in] evaluators - a list of evaluators
   */
  void DefineEvaluators(
      const std::list<Evaluator<NodeT,ScalarT>* >& evaluators);
  //      const ArrayRCP<Evaluator<NodeT,ScalarT> >& evaluators);
  
  /*!
   * \brief defines the size of the worksets
   * \param[in] num_pdes - number of PDEs = number of unknowns per DoF
   * \param[in] total_elems - total number of elements over all sets
   * \param[in] num_elems_per_set - number of elements in each work set
   */
  void ResizeSets(const int& num_pdes, const int& total_elems,
                  const int& num_elems_per_set);


  /*!
   * \brief copies the solution from a linear algebra object into soln_data_
   * \param[in] set_idx - index of the desired workset batch
   * \param[in] sol - current solution vector
   *
   * \warning This is not general enough in its current form.  It assumes that
   * the basis functions are 1-to-1 with the mesh nodes (and have the same local
   * index)
   */
  void CopySolution(const int& set_idx, const ArrayRCP<const double>& sol);

  /*!
   * \brief uses resid_data_ to fill in the linear-system objects
   * \param[in] set_idx - index of the desired workset batch
   * \param[in,out] rhs - the right-hand-side vector (view of a Tpetra vector)
   * \param[in,out] jacobian - the system matrix (view of a Tpetra matrix)
   *
   * \warning This is not general enough in its current form.  It assumes that
   * the basis functions are 1-to-1 with the mesh nodes (and have the same local
   * index)
   */
  void Assemble(const int& set_idx, const ArrayRCP<double>& rhs,
                const RCP<MatrixT>& jacobian);
  
  /*!
   * \brief fills the given stiffness matrix and right-hand-side vector
   * \param[in] mesh - mesh object to reference physical node locations
   * \param[in] sol - current solution vector (for nonlinear problems)
   * \param[out] rhs - the system right-hand-side vector
   * \param[out] jacobian - the system jacobian/stiffness matrix
   */
  void BuildSystem(const MeshT& mesh, const RCP<const VectorT>& sol,
                   const RCP<VectorT>& rhs, const RCP<MatrixT>& jacobian);
  
#if 0
  /*!
   * \brief copies the physical node coordinates from a mesh into the workset
   * \param[in] mesh - mesh object to reference physical node locations
   * \param[in] set_idx - the batch of elements whose nodes we want
   */
  void CopyMeshCoords(const MeshT& mesh, const int& set_idx);
#endif
  
 protected:
  int dim_; ///< spatial dimension that the fields are embedded in
  int num_pdes_; ///< number of unknowns per degree of freedom = # of PDEs
  int num_nodes_per_elem_; ///< number of nodes defining the elements
  int num_sets_; ///< number of work sets
  int num_elems_; ///< number of elements (volume, face, or line) per set
  int rem_num_elems_; ///< number of elements on last set
  int cub_dim_; ///< dimension of the cubature points (not necessarily = dim_)
  int num_cub_points_; ///< number of cubature points
  int num_ref_basis_; ///< number of basis functions on the reference element
  RCP<ostream> out_; ///< output stream
  RCP<CellTopology> topology_; ///< element topology
  RCP<const Basis<double, FieldContainer<double> >
      > basis_; ///< finite element basis on reference element
  //ArrayRCP<Evaluator<NodeT,ScalarT> > evaluators_; ///< graph of the evaluators
  std::list<Evaluator<NodeT,ScalarT>* > evaluators_; ///< graph of the evaluators 
  FieldContainer<double> cub_points_; ///< cubature point locations
  FieldContainer<double> cub_weights_; ///< cubature weights
  FieldContainer<double> vals_; ///< basis values at cub points on ref element
  FieldContainer<double> grads_; ///< gradient values at cub points ref element
  
  std::map<std::string,int> mesh_map_offset_; ///< map to mesh field data offset
  std::map<std::string,int> soln_map_offset_; ///< map to soln field data offset
  std::map<std::string,int> resid_map_offset_; ///< map to residual data offset
  ArrayRCP<NodeT> mesh_data_; ///< continuous array for node-based data
  ArrayRCP<ScalarT> soln_data_; ///< continuous array for solution data
  ArrayRCP<ResidT> resid_data_; ///< continuous array for output data
  ArrayRCP<typename
           MeshT::LocIdxT> index_data_; ///< continuous array for node indices
};

} // namespace davinci

// include the templated defintions
#include "work_set_def.hpp"

#endif
