output "vpc_id" {
  value = aws_vpc.main.id
  
}

output "instance_ip" {
  value = aws_instance.app.private_ip  
  sensitive = false  
}


output "subnet_ids" {
  value = aws_subnet.private.*.id  
}