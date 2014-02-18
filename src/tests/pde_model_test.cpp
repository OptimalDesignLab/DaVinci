/**
 * \file pde_model_test.cpp
 * \brief unit test for the PDEModel class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "pde_model.hpp"
#include "simple_mesh.hpp"
#include "work_set_factory.hpp"
#include "laplace.hpp"
#include "work_set.hpp"

using Teuchos::GlobalMPISession;
using Teuchos::RCP;
using Teuchos::ParameterList;
using Teuchos::Comm;
using Teuchos::DefaultComm;
using davinci::SimpleMesh;
using davinci::PDEModel;
using davinci::WorkSetFactoryBase;
using davinci::WorkSetFactory;
using davinci::LaplaceFactory;
using davinci::WorkSetBase;

BOOST_AUTO_TEST_SUITE(PDEModel_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  // create Teuchos communicator
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  // stream for output
  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (comm->getRank() == 0) ? std::cout : blackHole;
  // construct
  PDEModel<SimpleMesh> pde(out, comm);
}

BOOST_AUTO_TEST_CASE(NumPDEs_and_InitializeMesh) {
  // create Teuchos communicator
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  // stream for output
  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (comm->getRank() == 0) ? std::cout : blackHole;
  // construct pde with 5 equations
  PDEModel<SimpleMesh> pde(out, comm);
  pde.set_num_pdes(5);
  // initialize a rectangular mesh
  ParameterList mesh_param;
  mesh_param.set("Mesh Type","Rectangular");
  mesh_param.set("Nx",10);
  mesh_param.set("Ny",10);
  pde.InitializeMesh(mesh_param);
}

BOOST_AUTO_TEST_CASE(CreateMapAndJacobianGraph) {
  // create Teuchos communicator
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  // stream for output
  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (comm->getRank() == 0) ? std::cout : blackHole;
  // construct pde with 5 equations
  PDEModel<SimpleMesh> pde(out, comm);
  pde.set_num_pdes(5);
  // initialize a rectangular mesh
  ParameterList mesh_param;
  mesh_param.set("Mesh Type","Rectangular");
  mesh_param.set("Nx",10);
  mesh_param.set("Ny",10);
  pde.InitializeMesh(mesh_param);
  // build the map and graph
  pde.CreateMapAndJacobianGraph();
}

BOOST_AUTO_TEST_CASE(CreateLinearSystemWorkSets) {
  // create Teuchos communicator
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  // stream for output
  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (comm->getRank() == 0) ? std::cout : blackHole;
  // construct pde
  PDEModel<SimpleMesh> pde(out, comm);
  // initialize a rectangular mesh
  ParameterList mesh_param;
  mesh_param.set("Mesh Type","Rectangular");
  mesh_param.set("Nx",10);
  mesh_param.set("Ny",10);
  pde.InitializeMesh(mesh_param);
  // create a Laplace WorkSet factory and use it to create worksets
  RCP<WorkSetFactoryBase<SimpleMesh> > MyFactory = Teuchos::rcp(
      new WorkSetFactory<LaplaceFactory<SimpleMesh> >);
  Teuchos::ParameterList options;
  options.set("basis degree",1);
  options.set("cub degree",2);
  options.set("num elems per set", 10);
  pde.CreateLinearSystemWorkSets(options, MyFactory);
}

BOOST_AUTO_TEST_CASE(BuildLinearSystem) {
  // create Teuchos communicator
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  // stream for output
  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (comm->getRank() == 0) ? std::cout : blackHole;
  // construct pde
  PDEModel<SimpleMesh> pde(out, comm);
  // initialize a rectangular mesh
  ParameterList mesh_param;
  mesh_param.set("Mesh Type","Rectangular");
  mesh_param.set("Nx",10);
  mesh_param.set("Ny",10);
  pde.InitializeMesh(mesh_param);
  // create a Laplace WorkSet factory and use it to create worksets
  RCP<WorkSetFactoryBase<SimpleMesh> > MyFactory = Teuchos::rcp(
      new WorkSetFactory<LaplaceFactory<SimpleMesh> >);
  Teuchos::ParameterList options;
  options.set("basis degree",1);
  options.set("cub degree",2);
  options.set("num elems per set", 10);
  pde.CreateLinearSystemWorkSets(options, MyFactory);
  // build the Laplace linear system
  pde.CreateMapAndJacobianGraph();  
  pde.BuildLinearSystem();
}

  
BOOST_AUTO_TEST_SUITE_END()
