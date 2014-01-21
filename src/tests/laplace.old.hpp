/**
 * \file laplace.hpp
 * \brief declaration of a Laplace equation solver for testing
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_TESTS_LAPLACE_HPP
#define DAVINCI_SRC_TESTS_LAPLACE_HPP

#include "work_set.hpp"
#include "simple_mesh.hpp"

namespace davinci {

/*!
 * \class Laplace
 * \brief Equation set specialized to solve the Laplace equation
 *
 * This class is used primarly to test other base or template classes, and to
 * assess different stategies.
 */
class Laplace : public WorkSet<double, SimpleMesh> {  
 public:

  /*!
   * \brief constructor
   * \param[in] out - a valid output stream
   * \param[in] cell - pointer to shards cell topology attributes
   */
  Laplace(ostream& out);

  /*!
   * \brief defines the size of the workset fields
   * \param[in] total_elems - total number of elements over all sets
   * \param[in] num_elems_per_set - number of elements in each work set
   */
  void ResizeSets(const int& total_elems, const int& num_elems_per_set);

  /*!
   * \brief fills the given stiffness matrix and right-hand-side vector
   * \param[in] mesh - mesh object to reference physical node locations
   */
  void BuildSystem(const SimpleMesh& mesh);  
  
 protected:
  FieldContainer<double> local_stiff_matrix_; ///< elem. stiff matrices
  FieldContainer<double> vals_transformed_; ///< values at cubature points
  FieldContainer<double> vals_transformed_weighted_; ///< values weighted
  FieldContainer<double> grads_transformed_; ///< gradients in physical space
  FieldContainer<double> grads_transformed_weighted_; ///< weighted phys. grads.
};

} // namespace davinci

#endif
