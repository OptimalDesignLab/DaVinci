/**
 * \file evaluator.hpp
 * \brief class declaration for Evaluator
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_EVALUATOR_HPP
#define DAVINCI_SRC_COMMON_EVALUATOR_HPP

#include <boost/assert.hpp>
#include <boost/mpl/if.hpp>
#include "Teuchos_RCP.hpp"
#include "Teuchos_Tuple.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"

namespace davinci {

using std::ostream;
using std::map;
using std::string;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::Tuple;
using Intrepid::FieldContainer;
using shards::CellTopology;

/*!
 * \class Evaluator
 * \brief an abstract bass class for using data arrays to evaluate something
 * \tparam NodeT - the scalar type for node-based data (double, sacado AD, etc)
 * \tparam ScalarT - the scalar type for sol-based data (double, sacado AD, etc)
 *
 * An object derived from Evaluator should be used to compute a specific
 * quantity, or set of quantities, using a generic interface.  For example,
 * given an array of fields, the Evaluator will have knowledge of where the
 * inputs are stored (say rho, rho-u and e) and where the outputs are stored
 * (say pressure and speed of sound).  It will use this knowledge to take the
 * inputs and compute the outputs.
 *
 * \remark This class is a shameless imitation of the Evaluators used in
 * Phalanx, but "dumbed" down to make them more user friendly (hopefully).
 *
 * \remark There is no virtual destructor defined, which is sometimes considered
 * bad practice for an abstract base class; however, Evaluators should NEVER be
 * used to store dynamic data, since that is not their purpose.  Hence, memory
 * managment during object destruction should not be an issue.
 */
template <typename NodeT, typename ScalarT>
class Evaluator {

 protected:
  /*!
   * \typedef ResidT
   * \brief type used for fields dependent on both NodeT and ScalarT
   * 
   * A meta-template conditional determines which type to use for fields that
   * are composed of both solution-type (ScalarT) and mesh-type (NodeT) data;
   * for example, the gradient of the solution.  We always want to use an AD
   * type if either of ScalarT or NodeT is an AD type.  The approach implemented
   * here is not very general: basically, we check the size of the type, and
   * assume that AD types will be larger.
   */
  typedef typename boost::mpl::if_c< (sizeof(NodeT) <= sizeof(ScalarT)),
    ScalarT, NodeT>::type ResidT;
  
 public:

  /*!
   * \brief default constructor that defines the output and input dependencies
   */
  Evaluator() {}

  /*!
   * \brief defines the dimensions required for loops and arrary allocation
   * \param[in] num_elems - number of elements in a work set
   * \param[in] num_nodes_per_elem - number of nodes that define elements
   * \param[in] num_cub_points - total number of cubature points in work set
   * \param[in] num_ref_basis - number of basis functions on reference element
   * \param[in] spatial_dim - the spatial dimension of the problem
   * \param[in] num_pdes - number of PDEs and unknown fields
   */
  void SetDimensions(const int& num_elems, const int& num_nodes_per_elem,
                     const int& num_cub_points, const int& num_ref_basis,
                     const int& spatial_dim, const int& num_pdes = 1);
  
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
      int& resid_offset, map<string,int>& resid_map_offset) const = 0;

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
      ArrayRCP<ResidT>& resid_data, map<string,int>& resid_map_offset) = 0;
  
  /*!
   * \brief compute the dependent variables based on the independent ones
   * \param[in] toplogy - shards CellTopology of the elements in the workset
   * \param[in] cub_points - cubature points on reference element
   * \param[in] cub_weights - cubature weights on reference element
   * \param[in] basis_vals - the ref element basis values at the cubature points
   * \param[in] basis_grads - the basis gradient at the cubature points
   */  
  virtual void Evaluate(const RCP<CellTopology>& topology,
                        const FieldContainer<double>& cub_points,
                        const FieldContainer<double>& cub_weights,
                        const FieldContainer<double>& basis_vals,
                        const FieldContainer<double>& basis_grads) = 0;
  
 protected:  
  int num_elems_; ///< total number of elements in the workset
  int num_nodes_per_elem_; ///< number of nodes defining the element
  int num_cub_points_; ///< number of cubature points in each element
  int num_ref_basis_; ///< number of basis functions on reference element
  int dim_; ///< spatial dimension of the problem
  int num_pdes_; ///< number of PDEs and unknowns
};

/*!
 * \brief creates a multi-dimensional array view of 1-d arrary data
 * \relates Evaluator
 * \tparam T - type of data stored in the array
 * \tparam N - number of dimensions
 * \param[in] data - 1d array that we want a multi-dimensional view of
 * \param[in] offset - offset to beginning of desired data
 * \param[in] dimensions - the dimensions of each index of the multi-array
 * \returns multi-dimensional array view of the data
 */
template <typename T, int N>
RCP<FieldContainer<T> > GenerateView(
    ArrayRCP<T>& data, const int& offset, const Tuple<int,N>& dimensions);

} // namespace davinci

// include the templated defintions
#include "evaluator_def.hpp"

#endif
