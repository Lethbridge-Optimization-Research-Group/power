# Multiperiod Optimal Power Flow (MPOPF)

## Utilities

- [MatPower Graph Display](https://matpower.app/)
- [MatPower Description of Case Format](https://matpower.org/docs/ref/matpower5.0/caseformat.html)
<!-- - [Markdown Admonitions](https://documenter.juliadocs.org/stable/showcase/#Admonitions) -->

## Introduction

Welcome to the MPOPF project.
This Julia package provides tools for analyzing and optimizing power flow in electrical networks over multiple time periods.

> **ℹ️ Note:**  
> For a more indepth description of the project please visit our [Documentation](https://maxim-ciobanu.github.io/OPF/)

## Features

- Support for AC and DC power flow models
- Multi-period optimization
- Uncertainty modeling
- Linearization techniques
- Local search optimization
- Feasibility checking

# Design Diagram

```mermaid
classDiagram
         AbstractMPOPFModelFactory <|-- ACMPOPFModelFactory
         AbstractMPOPFModelFactory <|-- DCMPOPFModelFactory
   
         AbstractMPOPFModel <|-- MPOPFModel
         AbstractMPOPFModel <|-- MPOPFModelUncertainty
   
         AbstractMPOPFModelFactory : +create_model(factory AbstractMPOPFModelFactory, time_periods Int64, factors Vector~Float64~, ramping_cost Int64) MPOPFModel
         AbstractMPOPFModelFactory : +create_model(factory AbstractMPOPFModelFactory, scenarios Dict, time_periods Int64, factors Vector~Float64~, ramping_cost Int64) MPOPFModelUncertainty
         AbstractMPOPFModel : +optimize_model(model AbstractMPOPFModel)
   
         ACMPOPFModelFactory : +file_path String
         ACMPOPFModelFactory : +optimizer Type
         ACMPOPFModelFactory : +ACMPOPFModelFactory(file_path, optimizer)
   
         DCMPOPFModelFactory : +file_path String
         DCMPOPFModelFactory : +optimizer Type
         DCMPOPFModelFactory : +DCMPOPFModelFactory(file_path, optimizer)
   
         MPOPFModel : +model JuMP.Model
         MPOPFModel : +data Dict
         MPOPFModel : +time_periods Int64
         MPOPFModel : +factors Vector~Float64~
         MPOPFModel : +ramping_cost Int64
         MPOPFModel : +MPOPFModel(model, data, time_periods, factors, ramping_cost)
   
         MPOPFModelUncertainty : +model JuMP.Model
         MPOPFModelUncertainty : +data Dict
         MPOPFModelUncertainty : +scenarios Dict
         MPOPFModelUncertainty : +time_periods Int64
         MPOPFModelUncertainty : +factors Vector~Float64~
         MPOPFModelUncertainty : +ramping_cost Int64
         MPOPFModelUncertainty : +MPOPFModelUncertainty(model, data, scenarios, time_periods, factors, ramping cost)
   
         %% Operations
         class Operations {
           <<interface>>
           +set_model_variables!(AbstractMPOPFModel, AbstractMPOPFModelFactory)
           +set_model_objective_function!(AbstractMPOPFModel, AbstractMPOPFModelFactory)
           +set_model_constraints!(AbstractMPOPFModel, AbstractMPOPFModelFactory)
           +set_model_uncertainty_variables!(MPOPFModelUncertainty)
           +set_model_uncertainty_objective_function!(MPOPFModelUncertainty, AbstractMPOPFModelFactory)
           +set_model_uncertainty_constraints!(MPOPFModelUncertainty, AbstractMPOPFModelFactory)
         }
         AbstractMPOPFModelFactory --> Operations
         MPOPFModel --> Operations
         MPOPFModelUncertainty --> Operations
```
