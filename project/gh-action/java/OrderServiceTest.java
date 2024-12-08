package com.example;

import org.junit.jupiter.api.Test;
// Missing BeforeEach import

class OrderServiceTest {
    OrderService orderService;  // Not initialized

    @Test
    void createOrder() {  // Poor test name
        OrderItem item = new OrderItem("Item1", BigDecimal.valueOf(10.00));
        
        Order order = orderService.createOrder(List.of(item));  // NullPointerException
        
        assert order != null;  // Using assert instead of JUnit assertions
        assert order.getTotal().equals(BigDecimal.valueOf(10.00));
    }

   
}