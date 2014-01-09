/**
 * \file simple_mesh.hpp
 * \brief class declaration for SimpleMesh
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_SIMPLE_MESH_HPP
#define DAVINCI_SRC_COMMON_SIMPLE_MESH_HPP

#include <ostream>
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Intrepid_FieldContainer.hpp"

namespace davinci {

using std::ostream;
using Teuchos::RCP;
using Teuchos::ParameterList;
using Intrepid::FieldContainer;

#if 0
/*!
 * \struct SimpleMeshTypes
 * \brief types that are used by the SimpleMesh class
 *
 * These typedefs are needed by the MeshAPI base class.  Any class derived from
 * MeshAPI must also define a MeshTypes.
 */
struct SimpleMeshTypes {
  typedef int elem_idx_type_; ///< integer type used for elements
  typedef int node_idx_type_; ///< integer type used for nodes
  typedef int ref_idx_type_ ; ///< integer type used for reference nodes
};
#endif

/*!
 * \class SimpleMesh
 * \brief a basic mesh for unit testing
 *
 * This class can be used for unit testing of PDE models.  It should not be used
 * for "serious" applications.  Simplex elements are assumed (presently).
 */
class SimpleMesh { //: public MeshAPI<SimpleMesh, SimpleMeshTypes> {
 public:
  typedef int elem_idx_type_; ///< integer type used for elements
  typedef int node_idx_type_; ///< integer type used for nodes
  typedef int ref_idx_type_ ; ///< integer type used for reference nodes

  /*!
   * \brief default constructor
   * \param[in] out - a valid output stream
   */
  SimpleMesh(ostream& out);

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

  /*!
   * \brief Maps a reference node index in a given element to its global index
   * \param[in] elem - element/cell index
   * \param[in] ref_node - the reference node whose global index we want
   * \returns the global index of ref_node on elem
   *
   * Prefer to use the alias ElemToNode, defined in the base class.
   * TODO: try making this private after testing
   */
  const int& ElemToNode(const int& elem, const int& ref_node) const;
  
 private:  
  typedef double ScalarT; ///< should be a template?
  int dim_; ///< spatial dimension
  int num_nodes_; ///< number of nodes
  int num_elems_; ///< number of elements
  RCP<ostream> out_; ///< output stream
  FieldContainer<ScalarT> node_coord_; ///< (x,y) coordinates of nodes
  FieldContainer<int> node_type_; ///< is node on Bnd (1) or not (0)
  FieldContainer<int> elem_to_node_; ///< maps elem ref nodes to global nodes
};

// inlined members
inline const int& SimpleMesh::get_num_nodes() const { return num_nodes_; }
inline const int& SimpleMesh::get_num_elems() const { return num_elems_; }
inline const int& SimpleMesh::ElemToNode(
    const int& elem, const int& ref_node) const {
  return elem_to_node_(elem, ref_node); }
  
} // namespace davinci

#endif // DAVINCI_SRC_COMMON_SIMPLE_MESH_HPP
