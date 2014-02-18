/**
 * \file work_set_factory.hpp
 * \brief class declaration for WorkSetFactoryBase and WorkSetFactory
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#ifndef DAVINCI_SRC_COMMON_WORKSET_FACTORY_HPP
#define DAVINCI_SRC_COMMON_WORKSET_FACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_Basis.hpp"
#include "work_set.hpp"

namespace davinci {

using Teuchos::ParameterList;
using Teuchos::RCP;
using Teuchos::Array;
using Intrepid::Basis;
using Intrepid::FieldContainer;

/*!
 * \class WorkSetFactoryBase
 * \brief a base class for creating worksets with appropriate evaluators
 * \tparam MeshT - a generic class of mesh
 *
 * This acts as a common base class for polymorphic WorkSet factories.  WorkSet
 * factories are responsible for generating WorkSets for specific mesh
 * topologies and evaluators.  WorkSets are templated on ScalarT and NodeT,
 * whose AD variants depend on the number of degrees of freedom on a given
 * topology and the number of PDEs.  These same types (ScalarT and NodeT) are
 * used in the evaluators.  Consequently, a WorkSet and its evaluators must be
 * created at the same time.  This is the purpose of WorkSetFactory; however, a
 * user need only create an intermediate WorkSetFactory (the template parameter
 * DerivedFactory in WorkSetFactory below) that is templated on ScalarT and
 * NodeT, and then use this intermediate factory as the template parameter in
 * WorkSetFactory.  The Laplace equation example illustrates one of these
 * intermediate factories.
 */
template <typename MeshT>
class WorkSetFactoryBase {
 public:
  
  /*!
   * \brief the number of PDEs/dependent variables
   */
  virtual int NumPDEs() const = 0;

  /*!
   * \brief generates a workset of the appropriate type for PDE linear system
   * \param[in] p - a list of options needed for the factory   
   * \param[in] basis - Intrepid basis type used to create WorkSet
   * \param[out] worksets - a list of WorkSets
   */
  virtual void BuildLinearSystemWorkSet(
      const ParameterList& p,
      const RCP<const Basis<double, FieldContainer<double> > >& basis, 
      Array<RCP<WorkSetBase<MeshT> > >& worksets) {}
};

//template <typename MeshT, template <typename> class DerivedFactory>
template <class DerivedFactory>
class WorkSetFactory : public DerivedFactory {
 public:
  typedef typename DerivedFactory::MeshType MeshT;

  /*!
   * \brief generates a workset of the appropriate type for PDE linear system
   * \param[in] p - a list of options needed for the factory   
   * \param[in] basis - Intrepid basis type used to create WorkSet
   * \param[out] worksets - a list of WorkSets
   */
  void BuildLinearSystemWorkSet(
      const ParameterList& p,
      const RCP<const Basis<double, FieldContainer<double> > >& basis, 
      Array<RCP<WorkSetBase<MeshT> > >& worksets);

 private:
  
};

} // namespace davinci

// include the templated defintions
#include "work_set_factory_def.hpp"

#endif
