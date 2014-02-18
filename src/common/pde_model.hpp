/**
 * \file pde_model.hpp
 * \brief abstract base class for discretized PDE models
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_PDE_MODEL_HPP
#define DAVINCI_SRC_COMMON_PDE_MODEL_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_BlockCrsGraph_decl.hpp"
#include "Tpetra_BlockMultiVector_decl.hpp"
#include "Tpetra_VbrMatrix.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_Basis.hpp"
#include "model.hpp"
#include "work_set.hpp"
#include "work_set_factory.hpp"

namespace davinci {

using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Tpetra::BlockMap;
using Tpetra::BlockCrsGraph;

using Intrepid::Basis;

/*!
 * \class PDEModel
 * \brief Class for PDE models
 * \tparam MeshT - a generic class of mesh
 *
 * This templated class should be used to create classes for PDE
 * discretizations.
 */
template <typename MeshT>
class PDEModel : public Model {
 public:

  /*!
   * \brief default constructor
   * \param[in] out - a valid output stream
   * \param[in] comm - a communicator
   *
   * In practice, a AbstractFactory model will likely be used to create
   * PDEModels, so this constructor is primarily for testing
   */
  PDEModel(ostream& out, const RCP<const Comm<int> >& comm);

  /*!
   * \brief defines the number of partial differential equations
   * \param[in] num_pdes - number of PDEs
   */
  void set_num_pdes(const int& num_pdes);
  
  /*!
   * \brief build, read or otherwise create the mesh
   * \param[in] p - a list of options needed to initialize the desired mesh
   *
   * By using a Teuchos::ParameterList, we can pass in very general information
   * that can be used by the underlying MeshT
   */
  void InitializeMesh(ParameterList& p);

  /*!
   * \brief build the Tpetra Map and graph of the Jacobian
   */
  void CreateMapAndJacobianGraph();

  /*!
   * \brief Create appropriate WorkSets for a given factory
   * \param[in] workset_factory - builds the WorkSets given the topology
   * \param[in] degree - polynomial degree of the basis
   *
   * Creates WorkSet objects for use in building the linear system that solves a
   * PDE; thus, the solution scalar type is set to an Sacado AD type.
   */
  void BuildLinearSystemWorkSets(
      const RCP<WorkSetFactoryBase<MeshT> >& workset_factory,
      const int& degree);
  
  /*!
   * \brief set topology, cubature, and allocate memory for equation sets
   * \param[in] p - a list of options needed to initialize the equation sets
   *
   * By using a Teuchos::ParameterList, we can pass in very general information
   * that can be used by the underlying Equation type
   */
  //void InitializeEquationSet(ParameterList& p);
  
  /*!
   * \brief default destructor
   */
  ~PDEModel() {}

 private:
  // convenience typedefs
  typedef typename MeshT::LocIdxT LocIdxT;
  typedef typename MeshT::GlbIdxT GlbIdxT;
  typedef Intrepid::Basis<double, Intrepid::FieldContainer<double> > BasisT;
  typedef Tpetra::BlockMultiVector<double,LocIdxT,GlbIdxT> VectorT;
  typedef Tpetra::VbrMatrix<double,LocIdxT,GlbIdxT> MatrixT;

  int num_pdes_; ///< number of PDEs (number variables per node)
  RCP<ostream> out_; ///< output stream
  MeshT mesh_; ///< mesh object
  Array<RCP<WorkSetBase<MeshT> > > workset_; ///< array of equation work sets
  RCP<const BlockMap<LocIdxT,GlbIdxT> >
  map_; ///< Tpetra map object that indicates local nodes and their indices
  RCP<BlockCrsGraph<LocIdxT,GlbIdxT> >
  jac_graph_; ///< Tpetra graph for the Jacobian matrix
  RCP<VectorT> sol_; ///< solution stored as a Tpetra linear algebra object
  RCP<VectorT> rhs_; ///< linear-system right-hand-side
  RCP<MatrixT> jac_; ///< linear-system Jacobian matrix
};

} // namespace davinci

// include the templated defintions
#include "pde_model_def.hpp"

#endif // DAVINCI_SRC_COMMON_PDE_MODEL_HPP
