package com.example;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

public class Order {
    private final String id;
    private final List<OrderItem> items;
    private final BigDecimal total;
    private OrderStatus status;
    private final LocalDateTime createdAt;

    public Order(String id, List<OrderItem> items, BigDecimal total, OrderStatus status) {
        this.id = id;
        this.items = List.copyOf(items); // Immutable copy
        this.total = total;
        this.status = status;
        this.createdAt = LocalDateTime.now();
    }

    // Getters
    public String getId() { return id; }
    public List<OrderItem> getItems() { return List.copyOf(items); }
    public BigDecimal getTotal() { return total; }
    public OrderStatus getStatus() { return status; }
    public LocalDateTime getCreatedAt() { return createdAt; }

    // Setter for status only
    public void setStatus(OrderStatus status) {
        this.status = status;
    }
}