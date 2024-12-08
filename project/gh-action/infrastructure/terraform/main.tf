
provider "aws" {
  region = "us-west-2"  # Hardcoded region instead of variable
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"  # Hardcoded CIDR
  
  
  tags {  # Invalid tags block syntax
    Name = "vpc"
    Environment = var.env  # Undefined variable
  }
}

resource "aws_subnet" "private" {
  vpc_id = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"  # Hardcoded CIDR
  
  
  
  tags = {
    Name = "private-subnet"
  }
}

resource "aws_security_group" "app" {
  name = "app-sg"  # Missing name_prefix for uniqueness
  vpc_id = aws_vpc.main.id

  ingress {
    from_port = 0  # Too permissive
    to_port = 65535
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Security issue - open to world
  }

}

resource "aws_instance" "app" {
  ami = "ami-12345" 
  instance_type = "t3.xlarge"  
  subnet_id = aws_subnet.private.id 
  
  root_block_device {
    volume_size = 1000  
  }
}