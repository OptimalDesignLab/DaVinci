/**
 * \file work_set.hpp
 * \brief class declaration for WorkSet
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_WORK_SET_HPP
#define DAVINCI_SRC_COMMON_WORK_SET_HPP

#include "Teuchos_RCP.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid_FieldContainer.hpp"

namespace davinci {

using std::ostream;
using Teuchos::RCP;
using shards::CellTopology;
using Intrepid::FieldContainer;

/*!
 * \class WorkSet
 * \brief a collection of data and functions for common-type elements
 * \tparam ScalarT - the scalar type (float, double, sacado AD, etc)
 */
template <typename ScalarT>
class WorkSet {
 public:
  
  /*!
   * \brief constructor
   * \param[in] out - a valid output stream
   */
  WorkSet(ostream& out);

  /*!
   * \brief initializes the topology of all elements in this workset
   * \param[in] cell - pointer to shards cell topology attributes
   */
  void DefineTopology(const CellTopologyData* cell);
  
  /*!
   * \brief sets the cubature points and weights based on degree and topology_
   * \param[in] degree - highest polynomial degree cubature is exact
   */
  void DefineCubature(const int& degree);
  
  /*!
   * \brief defines the size of the workset fields
   * \param[in] num_elems - number of elements in each work set
   */
  virtual void ResizeFields(const int& num_elems);

 protected:  
  int num_sets_; ///< number of work sets
  int num_elems_; ///< number of elements per set
  int cub_dim_; ///< dimension of the cubature points
  int num_cub_points_; ///< number of cubature points
  RCP<ostream> out_; ///< output stream
  RCP<CellTopology> topology_; ///< element topology
  FieldContainer<ScalarT> cub_points_; ///< cubature point locations
  FieldContainer<ScalarT> cub_weights_; ///< cubature weights
  FieldContainer<ScalarT> tri_vals_; ///< basis values on ref element
  FieldContainer<ScalarT> tri_grads_; ///< gradient values on ref elem.
};

} // namespace davinci

// include the templated defintions
#include "work_set_def.hpp"

#endif
