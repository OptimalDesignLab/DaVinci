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
#include "Tpetra_BlockMultiVector_decl.hpp"
#include "Tpetra_VbrMatrix.hpp"
#include "model.hpp"
//#include "workset.hpp"

namespace davinci {

using Teuchos::ArrayRCP;
using Tpetra::BlockMap;
using Tpetra::BlockCrsGraph;

/*!
 * \class PDEModel
 * \brief Class for PDE models
 * \tparam MeshT - a generic class of mesh
 * \tparam Equation - a generic equation (workset) class
 *
 * This templated class should be used to create classes for PDE
 * discretizations.
 * 
 * This class is templated on MeshT for speed.  We could have included a
 * pointer to an abstract Mesh base class, but this would have introduced
 * dynamic polymorphism (which has a runtime overhead cost) on methods that need
 * to be fast.
 */
template <typename MeshT, typename Equation>
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
   * \brief build, read or otherwise create the mesh
   * \param[in] p - a list of options needed to initialize the desired mesh
   *
   * By using a Teuchos::ParameterList, we can pass in very general information
   * that can be used by the underlying MeshT
   */
  void InitializeMesh(ParameterList& p);

  /*!
   * \brief set topology, cubature, and allocate memory for equation sets
   * \param[in] p - a list of options needed to initialize the equation sets
   *
   * By using a Teuchos::ParameterList, we can pass in very general information
   * that can be used by the underlying Equation type
   */
  void InitializeEquationSet(ParameterList& p);
  
  /*!
   * \brief default destructor
   */
  ~PDEModel() {}

 private:
  // convenience typedefs
  typedef Sacado::Fad::SFad<double,3*num_pdes> AutoDiffT; // not general!!!
  typedef Tpetra::BlockMultiVector<double,MeshT::LocIdxT,MeshT::GlbIdxT> VectorT;
  typedef Tpetra::VbrMatrix<double,MeshT::LocIdxT,MeshT::GlbIdxT> MatrixT;
  typedef WorkSet<double,AutoDiffT,MeshT> WorkSetT;
  
  int num_sets_; ///< number of equation work sets
  MeshT mesh_; ///< mesh object
  //ArrayRCP<Equation> work_set_; ///< array of equation work sets
  //Equation work_set_;
  

  WorkSet<double,ADType,SimpleMesh> MyWorkSet(out);
  RCP<const BlockMap<MeshT::LocIdxT,MeshT::GlbIdxT> >
  map; ///< Tpetra map object that indicates local nodes and their indices
  RCP<BlockCrsGraph<MeshT::LocIdxT,MeshT::GlbIdxT> >
  jac_graph; ///< Tpetra graph for the Jacobian matrix
};

} // namespace davinci

// include the templated defintions
#include "pde_model_def.hpp"

#endif // DAVINCI_SRC_COMMON_PDE_MODEL_HPP
