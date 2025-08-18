
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "lunar-habitat-rl"
}

variable "compliance_requirements" {
  description = "Compliance requirements by region"
  type        = map(list(string))
  default = {
    "us-east-1"       = ["SOC2", "PCI-DSS"]
    "eu-west-1"       = ["GDPR", "SOC2"]
    "ap-southeast-1"  = ["PDPA", "SOC2"]
    "ca-central-1"    = ["PIPEDA", "SOC2"]
  }
}
