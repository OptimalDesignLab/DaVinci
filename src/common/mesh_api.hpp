/**
 * \file mesh_api.hpp
 * \brief class declaration for MeshAPI
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_MESH_API_HPP
#define DAVINCI_SRC_COMMON_MESH_API_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Tpetra_BlockMap.hpp"
#include "Tpetra_BlockCrsGraph_decl.hpp"
#include "work_set.hpp"
#include "work_set_factory.hpp"

namespace davinci {

using Teuchos::ParameterList;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Intrepid::FieldContainer;
using Intrepid::Basis;
using Tpetra::BlockMap;
using Tpetra::BlockCrsGraph;

/*!
 * \class MeshAPI
 * \brief an abstract base class for different mesh tools
 * \tparam LocalIndexT - integer type used for local indices
 * \tparam GlobalIndexT - integer type used for global indices
 *
 * This interface allows us to use different mesh libraries or mesh objects. For
 * example, we may want to use SCOREC tools for large-scale complex
 * applications and in-house tools for simple unit tests.
 *
 * \remark See the SimpleMesh class (simple_mesh.hpp) for a concrete example.
 *
 * \remark This class is templated on LocalIndexT and GlobalIndexT to account
 * for libraries that use long int or unsigned integers for indices.
 *
 * \remark We may want to consider using the Template Method pattern here (see
 * "Virtuality" in C/C++ Users Journal, 19(9), Sept 2001).
 */
template <typename LocalIndexT, typename GlobalIndexT> 
class MeshAPI {
 public:
  // convenience typedefs
  typedef MeshAPI MeshT;
  typedef LocalIndexT LocIdxT;
  typedef GlobalIndexT GlbIdxT;
  typedef Intrepid::Basis<double, FieldContainer<double> > BasisT;
  
  /*!
   * \brief virtual to ensure correct memory management in derived classes
   */
  virtual ~MeshAPI() {}

  /*!
   * \brief create or read-in the desired mesh
   * \param[in] p - a list of options needed to initialize the desired mesh
   */
  virtual void Initialize(ParameterList& p) = 0;

  /*
   * \brief create map that indicates range of local DOF
   * \param[in] num_pdes - number of PDEs/vars at each node (defines block size)
   * \param[out] map - pointer to Tpetra BlockMap
   */
  virtual void BuildTpetraMap(const int& num_pdes,
                              RCP<const BlockMap<LocIdxT, GlbIdxT> >& map)
      const = 0;
  
  /*!
   * \brief create the graph correpsonding to the Jacobian matrix
   * \param[in] map - Tpetra map
   * \param[out] jac_graph - graph for the Jacobian matrix
   */
  virtual void BuildMatrixGraph(
      const RCP<const BlockMap<LocIdxT, GlbIdxT> >& map,
      RCP<BlockCrsGraph<LocIdxT, GlbIdxT> >& jac_graph) const = 0;

#if 0
  /*!
   * \brief creates worksets corresponding to element topologies on mesh
   * \param[in] workset_factory - builds the WorkSets given the topology
   * \param[in] degree - polynomial degree of the basis
   * \param[out] workset - array of WorkSet base class
   *
   * Creates WorkSet objects for use in building the linear system that solves a
   * PDE; thus, the solution scalar type is set to an Sacado AD type.
   */
  virtual void BuildLinearSystemWorkSets(
      const RCP<WorkSetFactoryBase<MeshT> >& workset_factory,
      const int& degree, Array<RCP<WorkSetBase<MeshT> > >& worksets
                                         ) const = 0;
#endif

  /*!
   * \brief returns a list of the distinct intrepid bases
   * \param[in] degree - polynomial degree of the basis
   * \param[out] bases - the list of bases
   */
  virtual void GetIntrepidBases(const int& degree,
                                Array<RCP<const BasisT> >& bases) const = 0;
  
  /*!
   * \brief access to the number of nodes; local to this process if parallel
   */
  virtual const LocalIndexT& get_num_nodes() const = 0;

  /*!
   * \brief access to the number of elements; local to this process if parallel
   */
  virtual const LocalIndexT& get_num_elems() const = 0;
  
  /*!
   * \brief copy the coordinates of the nodes in the given workset of elements
   * \param[out] coords - array to store the coordinates in
   * \param[in] set_idx - the workset index we want the coordinates from
   * \param[in] num_elems_per_set - the number of elements in each (typical) set
   * \param[in] num_sets - the total number of sets
   *
   * Since most, if not all, Meshing libraries use double-type for the node
   * coordinates, this member function may need to be specialized to handle AD
   * types.
   */
  virtual void CopyElemNodeCoords(ArrayRCP<double>& coords,
                                  const int& set_idx,
                                  const LocalIndexT& num_elems_per_set,
                                  const int& num_sets) const = 0;

  /*
   * \brief returns the indices for the basis functions on each element
   * \param[out] dof_index - array of (local) basis function indices
   * \param[in] set_idx - the workset index we want the coordinates from
   * \param[in] num_elems_per_set - the number of elements in each (typical) set
   * \param[in] num_sets - the total number of sets
   */
  virtual void CopyElemDOFIndices(ArrayRCP<LocIdxT>& dof_index,
                                  const int& set_idx,
                                  const int& num_elems_per_set,
                                  const int& num_sets) const = 0;
};

} // namespace davinci

#endif // DAVINCI_SRC_COMMON_MESH_API_HPP
