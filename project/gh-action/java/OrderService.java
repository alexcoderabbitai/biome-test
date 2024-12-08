package com.example;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public class OrderService {
    private final Map<String, Order> orders = new HashMap<>();

    public Order createOrder(List<OrderItem> items) {
        if (items == null || items.isEmpty()) {
            throw new IllegalArgumentException("Order must contain at least one item");
        }

        BigDecimal total = items.stream()
            .map(OrderItem::getPrice)
            .reduce(BigDecimal.ZERO, BigDecimal::add);

        Order order = new Order(
            UUID.randomUUID().toString(),
            items,
            total,
            OrderStatus.PENDING
        );

        orders.put(order.getId(), order);
        return order;
    }

    public Optional<Order> getOrder(String orderId) {
        return Optional.ofNullable(orders.get(orderId));
    }

    public Order updateOrderStatus(String orderId, OrderStatus status) {
        return orders.computeIfPresent(orderId, (id, order) -> {
            order.setStatus(status);
            return order;
        });
    }
}