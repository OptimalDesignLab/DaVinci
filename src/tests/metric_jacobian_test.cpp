/**
 * \file metric_jacobian_test.cpp
 * \brief unit test for the MetricJacobian class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_RCP.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "metric_jacobian.hpp"

using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Intrepid::CellTools;
using Intrepid::FieldContainer;
using davinci::Evaluator;
using davinci::MetricJacobian;

BOOST_AUTO_TEST_SUITE(MetricJacobian_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  MetricJacobian<double,double> jacob;
}

BOOST_AUTO_TEST_CASE(SetDimensions) {
  Evaluator<double,double>* jacob = new MetricJacobian<double,double>();
  int num_elems = 10;
  int num_nodes_per_elem = 3;
  int num_cub_points = 10;
  int num_ref_basis = 3;
  int dim = 3;
  jacob->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                       num_ref_basis, dim);
}

BOOST_AUTO_TEST_CASE(MemoryRequired) {
  Evaluator<float,double>* jacob = new MetricJacobian<float,double>();
  int num_elems = 10;
  int num_nodes_per_elem = 3;
  int num_cub_points = 10;
  int num_ref_basis = 3;
  int dim = 2;
  jacob->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                       num_ref_basis, dim);
  int mesh_offset = 0;
  int soln_offset = 0;
  int resid_offset = 0;
  std::map<std::string,int> mesh_map_offset, soln_map_offset, resid_map_offset;
  jacob->MemoryRequired(mesh_offset, mesh_map_offset,
                        soln_offset, soln_map_offset,
                        resid_offset, resid_map_offset);
  BOOST_CHECK_EQUAL( mesh_map_offset["jacob"], 0);
  BOOST_CHECK_EQUAL( mesh_map_offset["jacob_inv"],
                     num_elems*num_cub_points*dim*dim);
  BOOST_CHECK_EQUAL( mesh_map_offset["jacob_det"],
                     2*num_elems*num_cub_points*dim*dim);
}

BOOST_AUTO_TEST_CASE(SetDataView) {
  Evaluator<double,double>* jacob = new MetricJacobian<double,double>();
  int num_elems = 10;
  int num_nodes_per_elem = 3;
  int num_cub_points = 10;
  int num_ref_basis = 3;
  int dim = 2;
  jacob->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                       num_ref_basis, dim);
  int mesh_offset = num_elems*num_nodes_per_elem*dim;
  int soln_offset = 0;
  int resid_offset = 0;
  std::map<std::string,int> mesh_map_offset, soln_map_offset, resid_map_offset;
  mesh_map_offset["node_coords"] = 0;
  jacob->MemoryRequired(mesh_offset, mesh_map_offset,
                        soln_offset, soln_map_offset,
                        resid_offset, resid_map_offset);
  Teuchos::ArrayRCP<double> mesh_data(mesh_offset, 1.0);
  Teuchos::ArrayRCP<double> soln_data(soln_offset, 1.0);
  Teuchos::ArrayRCP<typename davinci::Evaluator<double,double>::ResidT> resid_data(resid_offset, 1.0);
  jacob->SetDataViews(mesh_data, mesh_map_offset, soln_data, soln_map_offset,
                      resid_data, resid_map_offset);
}

BOOST_AUTO_TEST_CASE(SetReferenceElementData) {
  Evaluator<double,double>* jacob = new MetricJacobian<double,double>();
  int num_elems = 10;
  int dim = 2;
  // define a trianglular-element topology
  Teuchos::RCP<const CellTopologyData>
      cell(shards::getCellTopologyData<shards::Triangle<3> >(), false);
  RCP<shards::CellTopology> topology =
      Teuchos::rcp(new shards::CellTopology(cell.get()));
  int num_nodes_per_elem = topology->getNodeCount();
  Intrepid::DefaultCubatureFactory<double> cubFactory;
  int degree = 3;

  // define the cubature for the element
  RCP<Intrepid::Cubature<double> >
      cub = cubFactory.create(*topology, degree);
  int num_cub_points = cub->getNumPoints();
  FieldContainer<double> cub_points(num_cub_points, dim);
  FieldContainer<double> cub_weights(num_cub_points);
  cub->getCubature(cub_points, cub_weights);

  // define the cubature for the sides
  RCP<Intrepid::Cubature<double> > side_cub;
  int num_sides = topology->getSideCount();
  std::vector<int> side_cub_dim(num_sides);
  std::vector<int> side_num_cub_points(num_sides);
  ArrayRCP<FieldContainer<double> > side_cub_points(num_sides);
  ArrayRCP<FieldContainer<double> > side_cub_weights(num_sides);  
  for (unsigned si = 0; si < num_sides; si++) {
    shards::CellTopology side_topo(topology->getBaseCellTopologyData(dim-1,si));
    side_cub = cubFactory.create(side_topo, degree);
    side_cub_dim[si] = side_cub->getDimension();
    side_num_cub_points[si] = side_cub->getNumPoints();
    side_cub_points[si].resize(side_num_cub_points[si], side_cub_dim[si]);
    side_cub_weights[si].resize(side_num_cub_points[si]);
    side_cub->getCubature(side_cub_points[si], side_cub_weights[si]);
  }
  
  // define the FE basis on the cubature points
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double> > basis;
  int num_ref_basis = basis.getCardinality();
  FieldContainer<double> vals(num_ref_basis, num_cub_points);
  FieldContainer<double> grads(num_ref_basis, num_cub_points, dim);
  basis.getValues(vals, cub_points, Intrepid::OPERATOR_VALUE);
  basis.getValues(grads, cub_points, Intrepid::OPERATOR_GRAD);

  // define the FE basis on the side cubature points
  ArrayRCP<FieldContainer<double> > side_vals(num_sides);
  ArrayRCP<FieldContainer<double> > side_grads(num_sides);
  for (unsigned si = 0; si < num_sides; si++) {
    side_vals[si].resize(num_ref_basis, side_num_cub_points[si]);
    side_grads[si].resize(num_ref_basis, side_num_cub_points[si], dim);
    basis.getValues(side_vals[si], side_cub_points[si],
                    Intrepid::OPERATOR_VALUE);
    basis.getValues(side_grads[si], side_cub_points[si],
                    Intrepid::OPERATOR_GRAD);
  }

  jacob->SetReferenceElementData(*topology, cub_points, cub_weights, vals, grads,
                                 side_cub_points, side_cub_weights, side_vals,
                                 side_grads);
}

BOOST_AUTO_TEST_CASE(Evaluate) {
  Evaluator<double,double>* jacob = new MetricJacobian<double,double>();
  int num_elems = 10;
  int dim = 2;

  // define a trianglular-element topology and its cubature
  Teuchos::RCP<const CellTopologyData>
      cell(shards::getCellTopologyData<shards::Triangle<3> >(), false);
  RCP<shards::CellTopology> topology =
      Teuchos::rcp(new shards::CellTopology(cell.get()));
  int num_nodes_per_elem = topology->getNodeCount();
  Intrepid::DefaultCubatureFactory<double> cubFactory;
  int degree = 3;
  RCP<Intrepid::Cubature<double> >
      cub = cubFactory.create(*topology, degree);
  int num_cub_points = cub->getNumPoints();
  FieldContainer<double> cub_points(num_cub_points, dim);
  FieldContainer<double> cub_weights(num_cub_points);
  cub->getCubature(cub_points, cub_weights);

  // define the FE basis
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double> > basis;
  int num_ref_basis = basis.getCardinality();
  FieldContainer<double> vals(num_ref_basis, num_cub_points);
  FieldContainer<double> grads(num_ref_basis, num_cub_points, dim);
  basis.getValues(vals, cub_points, Intrepid::OPERATOR_VALUE);
  basis.getValues(grads, cub_points, Intrepid::OPERATOR_GRAD);

  // Initialize the MetricJacobian evaluator
  jacob->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                       num_ref_basis, dim);
  int mesh_offset = num_elems*num_nodes_per_elem*dim;
  int soln_offset = 0;
  int resid_offset = 0;
  std::map<std::string,int> mesh_map_offset, soln_map_offset, resid_map_offset;
  mesh_map_offset["node_coords"] = 0;
  jacob->MemoryRequired(mesh_offset, mesh_map_offset,
                        soln_offset, soln_map_offset,
                        resid_offset, resid_map_offset);
  Teuchos::ArrayRCP<double> mesh_data(mesh_offset, 1.0);
  Teuchos::ArrayRCP<double> soln_data(soln_offset, 1.0);
  Teuchos::ArrayRCP<double> resid_data(resid_offset, 1.0);
  jacob->SetDataViews(mesh_data, mesh_map_offset, soln_data, soln_map_offset,
                      resid_data, resid_map_offset);

  // define node coordinates; the elements do not need to be adjacent
  for (int i = 0; i < num_elems; i++) {
    // node 1
    mesh_data[i*num_nodes_per_elem*dim] = static_cast<double>(i); //< x
    mesh_data[i*num_nodes_per_elem*dim+1] = 0.0; //< y
    // node 2
    mesh_data[i*num_nodes_per_elem*dim+2] = static_cast<double>(i+1); //< x
    mesh_data[i*num_nodes_per_elem*dim+3] = 0.0; //< y
    // node 3
    mesh_data[i*num_nodes_per_elem*dim+4] = static_cast<double>(i); //< x
    mesh_data[i*num_nodes_per_elem*dim+5] = 1.0; //< y
  }
  jacob->Evaluate(*topology, cub_points, cub_weights, vals, grads);

  // check that Jacobian determinant is 1.0 at all cubature points
  for (int i = 0; i < num_elems; i++)
    for (int j = 0; j < num_cub_points; j++) {
      BOOST_CHECK_CLOSE(mesh_data[mesh_map_offset["jacob_det"]
                                  +i*num_cub_points+j], 1.0, 1e-13);
    }
}

BOOST_AUTO_TEST_CASE(Evaluate_Ver2) {
  Evaluator<double,double>* jacob = new MetricJacobian<double,double>();
  int num_elems = 10;
  int dim = 2;
  // define a trianglular-element topology
  Teuchos::RCP<const CellTopologyData>
      cell(shards::getCellTopologyData<shards::Triangle<3> >(), false);
  RCP<shards::CellTopology> topology =
      Teuchos::rcp(new shards::CellTopology(cell.get()));
  int num_nodes_per_elem = topology->getNodeCount();
  Intrepid::DefaultCubatureFactory<double> cubFactory;
  int degree = 3;

  // define the cubature for the element
  RCP<Intrepid::Cubature<double> >
      cub = cubFactory.create(*topology, degree);
  int num_cub_points = cub->getNumPoints();
  FieldContainer<double> cub_points(num_cub_points, dim);
  FieldContainer<double> cub_weights(num_cub_points);
  cub->getCubature(cub_points, cub_weights);

  // define the cubature for the sides
  RCP<Intrepid::Cubature<double> > side_cub;
  int num_sides = topology->getSideCount();
  std::vector<int> side_cub_dim(num_sides);
  std::vector<int> side_num_cub_points(num_sides);
  ArrayRCP<FieldContainer<double> > side_cub_points(num_sides);
  ArrayRCP<FieldContainer<double> > side_cub_weights(num_sides);  
  for (unsigned si = 0; si < num_sides; si++) {
    shards::CellTopology side_topo(topology->getBaseCellTopologyData(dim-1,si));
    side_cub = cubFactory.create(side_topo, degree);
    side_cub_dim[si] = side_cub->getDimension();
    side_num_cub_points[si] = side_cub->getNumPoints();
    side_cub_points[si].resize(side_num_cub_points[si], side_cub_dim[si]);
    side_cub_weights[si].resize(side_num_cub_points[si]);
    side_cub->getCubature(side_cub_points[si], side_cub_weights[si]);
  }
  
  // define the FE basis on the cubature points
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double> > basis;
  int num_ref_basis = basis.getCardinality();
  FieldContainer<double> vals(num_ref_basis, num_cub_points);
  FieldContainer<double> grads(num_ref_basis, num_cub_points, dim);
  basis.getValues(vals, cub_points, Intrepid::OPERATOR_VALUE);
  basis.getValues(grads, cub_points, Intrepid::OPERATOR_GRAD);

  // define the FE basis on the side cubature points
  ArrayRCP<FieldContainer<double> > side_vals(num_sides);
  ArrayRCP<FieldContainer<double> > side_grads(num_sides);
  for (unsigned si = 0; si < num_sides; si++) {
    side_vals[si].resize(num_ref_basis, side_num_cub_points[si]);
    side_grads[si].resize(num_ref_basis, side_num_cub_points[si], dim);
    basis.getValues(side_vals[si], side_cub_points[si],
                    Intrepid::OPERATOR_VALUE);
    basis.getValues(side_grads[si], side_cub_points[si],
                    Intrepid::OPERATOR_GRAD);
  }

  // Initialize the MetricJacobian evaluator
  jacob->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                       num_ref_basis, dim);
  jacob->SetReferenceElementData(*topology, cub_points, cub_weights, vals, grads,
                                 side_cub_points, side_cub_weights, side_vals,
                                 side_grads);
  int mesh_offset = num_elems*num_nodes_per_elem*dim;
  int soln_offset = 0;
  int resid_offset = 0;
  std::map<std::string,int> mesh_map_offset, soln_map_offset, resid_map_offset;
  mesh_map_offset["node_coords"] = 0;
  jacob->MemoryRequired(mesh_offset, mesh_map_offset,
                        soln_offset, soln_map_offset,
                        resid_offset, resid_map_offset);
  Teuchos::ArrayRCP<double> mesh_data(mesh_offset, 1.0);
  Teuchos::ArrayRCP<double> soln_data(soln_offset, 1.0);
  Teuchos::ArrayRCP<double> resid_data(resid_offset, 1.0);
  jacob->SetDataViews(mesh_data, mesh_map_offset, soln_data, soln_map_offset,
                      resid_data, resid_map_offset);

  // define node coordinates; the elements do not need to be adjacent
  for (int i = 0; i < num_elems; i++) {
    // node 1
    mesh_data[i*num_nodes_per_elem*dim] = static_cast<double>(i); //< x
    mesh_data[i*num_nodes_per_elem*dim+1] = 0.0; //< y
    // node 2
    mesh_data[i*num_nodes_per_elem*dim+2] = static_cast<double>(i+1); //< x
    mesh_data[i*num_nodes_per_elem*dim+3] = 0.0; //< y
    // node 3
    mesh_data[i*num_nodes_per_elem*dim+4] = static_cast<double>(i); //< x
    mesh_data[i*num_nodes_per_elem*dim+5] = 1.0; //< y
  }
  
  jacob->Evaluate();

  // check that Jacobian determinant is 1.0 at all cubature points
  for (int i = 0; i < num_elems; i++)
    for (int j = 0; j < num_cub_points; j++) {
      BOOST_CHECK_CLOSE(mesh_data[mesh_map_offset["jacob_det"]
                                  +i*num_cub_points+j], 1.0, 1e-13);
    }
}

BOOST_AUTO_TEST_SUITE_END()
