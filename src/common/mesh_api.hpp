/**
 * \file mesh_api.hpp
 * \brief class declaration for MeshAPI
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_MESH_API_HPP
#define DAVINCI_SRC_COMMON_MESH_API_HPP

#include "Teuchos_ParameterList.hpp"
#include "Intrepid_FieldContainer.hpp"

namespace davinci {

using Teuchos::ParameterList;
using Intrepid::FieldContainer;

/*!
 * \class MeshAPI
 * \brief an abstract base class for different mesh tools
 * \tparam NodeIntT - integer type used for node indices
 * \tparam ElemIntT - integer type used for element indices
 *
 * This interface allows us to use different mesh libraries or mesh objects. For
 * example, we may want to use SCOREC tools for large-scale complex
 * applications and in-house tools for simple unit tests.
 *
 * \remark See the SimpleMesh class (simple_mesh.hpp) for a concrete example.
 *
 * \remark This class is templated on NodeIntT and ElemIntT to account for
 * libraries that use long int or unsigned integers for indices.
 *
 * \remark We may want to consider using the Template Method pattern here (see
 * "Virtuality" in C/C++ Users Journal, 19(9), Sept 2001).
 */
template <typename NodeIntT, typename ElemIntT> 
class MeshAPI {
 public:
  typedef NodeIntT node_idx_type_;
  typedef ElemIntT elem_idx_type_;
  
  /*!
   * \brief virtual to ensure correct memory management in derived classes
   */
  virtual ~MeshAPI() {}

  /*!
   * \brief create or read-in the desired mesh
   * \param[in] p - a list of options needed to initialize the desired mesh
   */
  virtual void Initialize(ParameterList& p) = 0;

  /*!
   * \brief access to the number of nodes; local to this process if parallel
   */
  virtual const NodeIntT& get_num_nodes() const = 0;

  /*!
   * \brief access to the number of elements; local to this process if parallel
   */
  virtual const ElemIntT& get_num_elems() const = 0;
  
  /*!
   * \brief copy the coordinates of the nodes in the given workset of elements
   * \param[out] coords - array to store the coordinates in
   * \param[in] set_idx - the workset index we want the coordinates from
   * \param[in] num_elems_per_set - the number of elements in each (typical) set
   * \param[in] num_sets - the total number of sets
   *
   * The FieldContainer is specialized to a double type, because most (if not
   * all) mesh libraries will use double.
   */
  virtual void CopyElemNodeCoords(FieldContainer<double>& coords,
                                  const int& set_idx,
                                  const ElemIntT& num_elems_per_set,
                                  const int& num_sets) const = 0;
};

} // namespace davinci

#endif // DAVINCI_SRC_COMMON_MESH_API_HPP
