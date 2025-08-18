
# Global Infrastructure Configuration
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Multi-region provider configuration
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
}

provider "aws" {
  alias  = "ca_central_1"
  region = "ca-central-1"
}

# Global resources
resource "aws_route53_zone" "main" {
  name = "lunar-habitat-rl.com"
  
  tags = {
    Environment = "production"
    Service     = "lunar-habitat-rl"
    Compliance  = "global"
  }
}

# Multi-region EKS clusters
module "eks_us_east_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.us_east_1
  }
  
  cluster_name = "lunar-habitat-rl-us-east-1"
  region       = "us-east-1"
  node_groups = {
    main = {
      instance_types = ["m5.large", "m5.xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
  }
  
  tags = {
    Environment = "production"
    Region      = "primary"
    Compliance  = "SOC2,PCI-DSS"
  }
}

module "eks_eu_west_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  cluster_name = "lunar-habitat-rl-eu-west-1"
  region       = "eu-west-1"
  node_groups = {
    main = {
      instance_types = ["m5.large", "m5.xlarge"]
      min_size       = 2
      max_size       = 8
      desired_size   = 2
    }
  }
  
  tags = {
    Environment = "production"
    Region      = "secondary"
    Compliance  = "GDPR,SOC2"
  }
}

# Global CloudFront distribution
resource "aws_cloudfront_distribution" "global" {
  origin {
    domain_name = aws_route53_zone.main.name
    origin_id   = "lunar-habitat-rl-global"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "lunar-habitat-rl-global"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }
  
  price_class = "PriceClass_All"
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
  
  tags = {
    Name        = "lunar-habitat-rl-global-cdn"
    Environment = "production"
  }
}
