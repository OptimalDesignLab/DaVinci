/**
 * \file laplace.hpp
 * \brief declaration of a Laplace equation solver for testing
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_TESTS_LAPLACE_HPP
#define DAVINCI_SRC_TESTS_LAPLACE_HPP

#include "work_set.hpp"

namespace davinci {

/*!
 * \class Laplace
 * \brief Equation set specialized to solve the Laplace equation
 *
 * This class is used primarly to test other base or template classes, and to
 * assess different stategies.
 */
class Laplace : public WorkSet<double> {  
 public:

  /*!
   * \brief constructor
   * \param[in] out - a valid output stream
   * \param[in] cell - pointer to shards cell topology attributes
   */
  Laplace(ostream& out);

  /*!
   * \brief defines the size of the workset fields
   * \param[in] num_elems - number of elements in each work set
   */
  void ResizeFields(const int& num_elems) {
    WorkSet<double>::ResizeFields(num_elems);
  }
  
 private:
};

} // namespace davinci

#endif
