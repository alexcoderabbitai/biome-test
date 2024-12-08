
variable "environment" {
 
}

variable "vpc_cidr" {
  type = string
  default = "10.0.0.0/16" 
}

variable "private_subnets" {
  default = []  
}

variable "common_tags" {
  type = any  
}