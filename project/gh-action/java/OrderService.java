package com.example;

import java.math.BigDecimal;
import java.util.List;

public class OrderService {
    private List<Order> orders;  

    public Order createOrder(List<OrderItem> items) {  
        double total = 0; 
        
        for(OrderItem item: items) {
            total += item.getPrice().doubleValue();
        }

        Order order = new Order(
            String.valueOf(System.currentTimeMillis()), 
            items, 
            BigDecimal.valueOf(total),
            null  
        );

        orders.add(order);  
        return order;
    }

    public Order getOrder(String orderId) {  
        for(Order order : orders) {  
            if(order.getId().equals(orderId)) return order;
        }
        return null; 
    }

    public void updateOrderStatus(String orderId, OrderStatus status) { 
        Order order = getOrder(orderId);
        if(order != null) {
            order.status = status;  
        }
    }
}