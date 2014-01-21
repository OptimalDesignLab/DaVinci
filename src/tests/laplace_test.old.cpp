/**
 * \file laplace_test.cpp
 * \brief unit test for the PDEModel<SimpleMesh,Laplace> specialization
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_DefaultComm.hpp"
#include <Teuchos_Time.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_Basis.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "simple_mesh.hpp"
#include "laplace.hpp"
#include "pde_model.hpp"

using Teuchos::GlobalMPISession;
using Teuchos::RCP;
using Teuchos::ParameterList;
using Teuchos::Comm;
using Teuchos::DefaultComm;
using Intrepid::FieldContainer;
using Intrepid::Basis;
using davinci::SimpleMesh;
using davinci::Laplace;
using davinci::PDEModel;

typedef double ScalarT;

BOOST_AUTO_TEST_SUITE(Laplace_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  // Note: GlobalMPISession requires argc and argv, but these are hidden by
  // boost test; they can be accessed as shown below.
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm = DefaultComm<int>::getComm();
  PDEModel<SimpleMesh,Laplace> LaplacePDE(std::cout, comm);
}

BOOST_AUTO_TEST_CASE(Initialize) {
  // Note: GlobalMPISession requires argc and argv, but these are hidden by
  // boost test; they can be accessed as shown below.
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm = DefaultComm<int>::getComm();
  PDEModel<SimpleMesh,Laplace> LaplacePDE(std::cout, comm);
  ParameterList p;
  p.set("Mesh Type", "Rectangular");
  LaplacePDE.InitializeMesh(p);
  p.set("Topology", shards::getCellTopologyData<shards::Triangle<3> >());
  p.set("Cubature Degree", 2);
  RCP<const Intrepid::Basis_HGRAD_TRI_C1_FEM<ScalarT, FieldContainer<ScalarT> > > tri_hgrad_basis;
  p.set("Basis", tri_hgrad_basis);
  LaplacePDE.InitializeEquationSet(p);
}

BOOST_AUTO_TEST_SUITE_END()
