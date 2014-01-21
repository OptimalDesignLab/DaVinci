/**
 * \file laplace.hpp
 * \brief definitions for Laplace class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include "Intrepid_CellTools.hpp"
#include "laplace.hpp"

namespace davinci {
//==============================================================================
Laplace::Laplace(ostream& out) :
    WorkSet(out) {
}
//==============================================================================
void Laplace::ResizeSets(const int& total_elems, const int& num_elems_per_set) {
  // check on total_elems and num_elems_per_set performed in WorkSet::ResizeSets
  WorkSet<double,SimpleMesh>::ResizeSets(total_elems, num_elems_per_set);  
  vals_transformed_.resize(num_elems_, num_ref_basis_, num_cub_points_);
  vals_transformed_weighted_.resize(num_elems_, num_ref_basis_, num_cub_points_); 
  local_stiff_matrix_.resize(num_elems_, num_ref_basis_, num_ref_basis_);
  grads_transformed_.resize(num_elems_, num_ref_basis_, num_cub_points_, dim_);
  grads_transformed_weighted_.resize(num_elems_, num_ref_basis_,
                                     num_cub_points_, dim_);
}
//==============================================================================
void Laplace::BuildSystem(const SimpleMesh& mesh) {
  typedef Intrepid::CellTools<double> CellTools;
  typedef Intrepid::FunctionSpaceTools FST;  
  // loop over batches
  for (int bi = 0; bi < num_sets_; bi++) {
    // copy the mesh nodes for batch bi into the work set
    mesh.CopyElemNodeCoords(node_coords_, bi, num_elems_, num_sets_);
    //CopyMeshCoords(mesh, bi);

    // transform integration points to physical points
    //CellTools::mapToPhysicalFrame(phys_cub_points, cub_points_, nodes_coords_,
    //                              tri_topo_);
    
    // Compute cell Jacobians, their inverses and their determinants
    CellTools::setJacobian(jacob_, cub_points_, node_coords_, *topology_);
    CellTools::setJacobianInv(jacob_inv_, jacob_);
    CellTools::setJacobianDet(jacob_det_, jacob_);

#if 0
    // ************ Compute element HGrad stiffness matrices ********************

    // evaluate diffusivity function at physical points
    for (int nPt = 0; nPt < num_cub_points_; nPt++) {
      ScalarT x = phys_cub_points(0,nPt,0);
      ScalarT y = phys_cub_points(0,nPt,1);
      diff_data(0,nPt) = diff_(x, y);
    }
    
    // transform to physical coordinates
    FST::HGRADtransformGRAD<ScalarT>(tri_grads_transformed, tri_jacob_inv,
                                     tri_grads_);
    
    // compute weighted measure
    FST::computeCellMeasure<ScalarT>(weighted_measure, tri_jacob_det,
                                     cub_weights_);

    // multiply values with weighted measure
    FST::multiplyMeasure<ScalarT>(tri_grads_transformed_weighted,
                                 weighted_measure, tri_grads_transformed);

    // scale by diffusivity
    FST::scalarMultiplyDataField<ScalarT>(tri_grads_transformed, diff_data,
                                          tri_grads_transformed);
    
    // integrate to compute element stiffness matrix
    FST::integrate<ScalarT>(local_stiff_matrix, tri_grads_transformed,
                            tri_grads_transformed_weighted, COMP_BLAS);

    // assemble into global matrix
    for (int row = 0; row < num_ref_basis_; row++){
      for (int col = 0; col < num_ref_basis_; col++){
        int rowIndex = elem_to_node_(k,row);
        int colIndex = elem_to_node_(k,col);
        ScalarT val = local_stiff_matrix(0,row,col);
        stiff_matrix_->InsertGlobalValues(1, &rowIndex, 1, &colIndex, &val);
      }
    }

    // *********************** Build right hand side ****************************

    // evaluate right hand side function at physical points
    for (int nPt = 0; nPt < num_cub_points_; nPt++){
      ScalarT x = phys_cub_points(0,nPt,0);
      ScalarT y = phys_cub_points(0,nPt,1);
      rhs_data(0,nPt) = divgrad_exact_(x, y);
    }

    // transform basis values to physical coordinates
    FST::HGRADtransformVALUE<ScalarT>(tri_vals_transformed, tri_vals_);

    // multiply values with weighted measure
    FST::multiplyMeasure<ScalarT>(tri_vals_transformed_weighted,
                                  weighted_measure, tri_vals_transformed);

    // integrate rhs term
    FST::integrate<ScalarT>(local_rhs, rhs_data, tri_vals_transformed_weighted,
                           COMP_BLAS);

    // assemble into global vector
    for (int row = 0; row < num_ref_basis_; row++){
      int rowIndex = elem_to_node_(k,row);
      ScalarT val = -local_rhs(0,row);
      rhs_->SumIntoGlobalValues(1, &rowIndex, &val);
    }
#endif 

  }
  
  
}


//==============================================================================
} // namespace davinci
