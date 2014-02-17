/**
 * \file work_set_factory_def.hpp
 * \brief definition WorkSetFactory methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include "evaluator.hpp"

namespace davinci {
//==============================================================================
//template <typename MeshT, template <typename> class DerivedFactory>
template <class DerivedFactory>
void WorkSetFactory<DerivedFactory>::BuildLinearSystemWorkSet(
    const RCP<const Basis<double, FieldContainer<double> > >& basis, 
    Array<RCP<WorkSetBase<MeshT> > >& worksets) {

  // determine the number of AD dependent variables
  const int num_autodiff_dep_vars = basis->getCardinality()*(
      this->num_pdes_);
  switch (num_autodiff_dep_vars) {
    case (2) : {
      // linear (p=1) line elements
      typedef double NodeT;
      typedef Sacado::Fad::SFad<double,2> ScalarT;
      Array<RCP<Evaluator<NodeT,ScalarT> > > evaluators;
      DerivedFactory::template CreateEvaluators<NodeT,ScalarT>(evaluators);
      worksets.push_back(Teuchos::rcp(new WorkSet<NodeT,ScalarT,MeshT>(
          basis, evaluators)));
      break;
    }
    case (3) : {
      // e.g. linear (p=1) triangles, or quadratic line elements
      typedef double NodeT;
      typedef Sacado::Fad::SFad<double,3> ScalarT;
      Array<RCP<Evaluator<NodeT,ScalarT> > > evaluators;
      DerivedFactory::template CreateEvaluators<NodeT,ScalarT>(evaluators);
      worksets.push_back(Teuchos::rcp(new WorkSet<NodeT,ScalarT,MeshT>(
          basis, evaluators)));
      break;
    }
    default : {
      std::cerr << "WorkSetFactory::BuildLinearSystemWorkSet(): "
                << "unrecognized num_autodiff_dep_vars..."
                << "may need to generalize this member function\n";
      throw(-1);
    }
  }
}
//==============================================================================
} // namespace davinci
