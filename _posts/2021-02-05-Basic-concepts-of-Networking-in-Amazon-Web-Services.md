I recently took a course on Pluralsight called AWS developer: getting started.

The course is simply about deploying a website on aws "properly".

Here properly means setting up a virtual private cloud, subnets, security groups, and a load balancer
with multiple EC2 instances available.

Here I'm going to note down a few concepts important in the AWS networking ecosystem, just for reminding myself.

**VPC**: A virtual private cloud (VPC) is an on-demand configurable pool of shared resources allocated within a public cloud environment, providing a certain level of isolation between the different users of the cloud using the resources. The isolation between one VPC user and all other users of the same cloud is achieved normally through allocation of a private IP subnet and a virtual communication construct (could be encrypted communication channels) per user. 


So the virtual private cloud is used to seperate resources from the rest of the cloud, and within this VPC we can define which resources can communicate and which resources are connected to the internet and which are private. When creating a VPC we define a range of IP addresses used within that VPC. VPC uses routing tables which declares attempted destination IPs and where they should be routed to - this can be used if you want to run all outgoing traffic through a proxy. Furthermore a VPC contains a Network access control list (NACL) which acts like a IP filter saying which IPs are allowed for both incoming and outgoing traffic.


**subnets**: an additional area within a VPC that has its own CIDR block (so a subrange of the IP adresses we defined for the VPC). It also has its own routing table and access control list. We can create a public subnet that can access the internet and a private subnet that's not accessible from the internet and must go through a NAT(network address translation) gateway to access the internet. A resource must always be assigned to a subnet.


**security group**: defines a set of allowed incoming/outgoing IP addresses and ports, security groups are attached at the instance level and multiple instances can belong to one security group, thus sharing allowed IP addresses. You can set security groups to allow access from other security groups instead of by IP. Security groups are kind of like conviently packaged Network access control lists (NACL)


**custom Amazon machine Instance**: custom AMI let's you take a snapshot of an EC2 instance including all the installed software. This enables us to reproduce a customized EC2 instance.

**NAT gateway**: network adress tranlation gateway is used to allow instances in a private subnet to send requests to the internet but still prevent the internet to initiate connections to the private instances. The NAT gateway itself must however live in a public subnet. So the instance in the private subnet sends a request to the NAT gateway which changes the source IP of the request to the IP of the NAT gateway, and sends the request forward so the receiver of this request never even sees the IP of the private instance. Once the NAT receives a response it forwards it to the private instance.



**Auto scaling group**: instead of monitoring traffic and manually creating new EC2 instances we can use Auto-scaling groups which use a launch template and scaling rules to expand or shrink a pool of instances automatically. The launch template  defines the image used, subnet, shell script when launching the image, maximum and minimum number of images and scaling rules (i.e. when to launch new instances).  We can set up a load balancer to route incoming requests to the EC2 instances in the autoscaling group. By that we have a stable DNS entrypoint and we can balance requests to multiple instances in the autoscaling group.


