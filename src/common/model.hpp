/**
 * \file model.hpp
 * \brief abstract base class for disciplinary analysis
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_MODEL_HPP
#define DAVINCI_SRC_COMMON_MODEL_HPP

#include <ostream>
#include "Teuchos_RCP.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_ParameterList.hpp"

namespace davinci {

using std::ostream;
using Teuchos::RCP;
using Teuchos::Comm;
using Teuchos::ParameterList;

/*!
 * \class Model
 * \brief abstract base class for disciplinary analysis
 *
 * This abstract base class should be used to derive concrete (or abstract)
 * classes for different disciplines.  This class should be sufficiently general
 * that it can be used for PDE-based models, ODE-based models, or algebraic
 * models.
 *
 * Developer notes: You may add member function to this class, but DO NOT
 * introduce member functions that are specific to a particular model.  For
 * example, we do not want a BuildMesh() method here, because some models will
 * not need a mesh.  Also, please AVOID adding data members: even including out_
 * and comm_ is not good practice.
 */
class Model {
 public:

  /*!
   * \brief default constructor
   * \param[in] out - a valid output stream
   * \param[in] comm - an abstract parallel communicator
   */
  explicit Model(ostream& out, const RCP<const Comm<int> >& comm) :
      comm_(comm), out_(&out, false) {}
  
  /*!
   * \brief default destructor
   */
  virtual ~Model() {}

 protected:
  RCP<ostream> out_; ///< output stream
  RCP<const Comm<int> > comm_; ///< Interface for dist. memory communication
};

} // namespace davinci

#endif // DAVINCI_SRC_COMMON_MODEL_HPP
