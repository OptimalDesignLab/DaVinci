/**
 * \file simple_mesh.hpp
 * \brief class declaration for SimpleMesh
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_SIMPLE_MESH_HPP
#define DAVINCI_SRC_COMMON_SIMPLE_MESH_HPP

#include <ostream>
#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Comm.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "mesh_api.hpp"

namespace davinci {

using std::ostream;
using Teuchos::Comm;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::ParameterList;
using Intrepid::FieldContainer;

/*!
 * \class SimpleMesh
 * \brief a basic mesh for unit testing
 *
 * This class can be used for unit testing of PDE models.  It should not be used
 * for "serious" applications.  Simplex elements are assumed (presently).
 */
class SimpleMesh : public MeshAPI<int,int> {
 public:

  /*!
   * \brief default constructor
   * \param[in] out - a valid output stream
   * \param[in] comm - an abstract parallel communicator
   */
  SimpleMesh(ostream& out, const RCP<const Comm<int> >& comm);

  /*!
   * \brief default destructor
   */
  ~SimpleMesh() {}

  /*!
   * \brief access to the number of nodes (locally)
   */
  const int& get_num_nodes() const;

  /*!
   * \brief access to the number of elements (locally)
   */
  const int& get_num_elems() const;
  
  /*!
   * \brief create or read-in the desired mesh
   * \param[in] p - a list of options needed to initialize the desired mesh
   */
  void Initialize(ParameterList& p);

  /*!
   * \brief creates a mesh consisting of triangles for the domain [0,Lx] X [0,Ly]
   * \param[in] Lx - length of the domain in the x-direction
   * \param[in] Ly - length of the domain in the y-direction
   * \param[in] Nx - number of intervals in the x-direction
   * \param[in] Ny - number of intervals in the y-direction
   */
  void BuildRectangularMesh(const double& Lx, const double& Ly, const int & Nx,
                            const int & Ny);

  /*
   * \brief create map that indicates range of local DOF
   * \param[in] map - Tpetra map
   */
  void BuildTpetraMap(RCP<const Map<LocIdxT,GlbIdxT> >& map) const;
  
  /*!
   * \brief create the graph correpsonding to the Jacobian matrix
   * \param[in] map - Tpetra map
   * \param[out] jac_graph - graph for the Jacobian matrix
   */
  void BuildMatrixGraph(
      const RCP<const Map<LocIdxT,GlbIdxT> >& map,
      RCP<CrsGraph<LocIdxT,GlbIdxT> >& jac_graph) const;
  
  /*!
   * \brief Maps a reference node index in a given element to its global index
   * \param[in] elem - element/cell index
   * \param[in] ref_node - the reference node whose global index we want
   * \returns the global index of ref_node on elem
   */
  const int& ElemToNode(const int& elem, const int& ref_node) const;

  /*!
   * \brief access to coordinate of a given element's node
   * \param[in] elem - element/cell index
   * \param[in] ref_node - the reference node whose global index we want
   * \param[in] dim - spatial dimension that we want
   * \returns x[dim] where x is the coordinate of ref_node on elem
   */
  const double& ElemNodeCoord(const int& elem, const int& ref_node,
                              const int& dim) const;

  /*!
   * \brief copy the coordinates of the nodes in the given workset of elements
   * \param[out] coords - array to store the coordinates in
   * \param[in] set_idx - the workset index we want the coordinates from
   * \param[in] num_elems_per_set - the number of elements in each (typical) set
   * \param[in] num_sets - the total number of sets
   */
  void CopyElemNodeCoords(ArrayRCP<double>& coords, const int& set_idx,
                          const int& num_elems_per_set,
                          const int& num_sets) const;

  /*
   * \brief returns the indices for the unknowns on each element
   * \param[out] dof_index - array of unknown variables' indices
   * \param[in] set_idx - the workset index we want the coordinates from
   * \param[in] num_elems_per_set - the number of elements in each (typical) set
   * \param[in] num_sets - the total number of sets
   * \param[in] num_pdes - number of equations
   *
   * \warning Presently, this expects the mesh vertices to be the only nodal
   * degrees of freedom, so no high order solutions at this time.
   */
  void CopyElemDOFIndices(ArrayRCP<LocIdxT>& dof_index, const int& set_idx,
                          const int& num_elems_per_set, const int& num_sets,
                          const int& num_pdes) const;
  
 private:
  int dim_; ///< spatial dimension
  int num_nodes_; ///< number of nodes
  int num_elems_; ///< number of elements
  RCP<ostream> out_; ///< output stream
  RCP<const Comm<int> > comm_; ///< interface for dist. memory communication
  FieldContainer<int> index_; ///< local node indices
  FieldContainer<int> global_index_; ///< global node indices
  FieldContainer<double> node_coord_; ///< (x,y) coordinates of nodes
  FieldContainer<int> node_type_; ///< is node on Bnd (1) or not (0)
  FieldContainer<int> elem_to_node_; ///< maps elem ref nodes to global nodes
};

// inlined members
inline const int& SimpleMesh::get_num_nodes() const { return num_nodes_; }
inline const int& SimpleMesh::get_num_elems() const { return num_elems_; }
inline const int& SimpleMesh::ElemToNode(
    const int& elem, const int& ref_node) const {
  return elem_to_node_(elem, ref_node); }
inline const double& SimpleMesh::ElemNodeCoord(
    const int& elem, const int& ref_node, const int& dim) const {
  node_coord_(elem_to_node_(elem, ref_node), dim); }
  
} // namespace davinci

#endif // DAVINCI_SRC_COMMON_SIMPLE_MESH_HPP
